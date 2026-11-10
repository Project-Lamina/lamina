use super::*;

const MAX_COPY_PIECES: usize = 4096;

fn push_leaves(
    ty: &IRType<'_>,
    base: u64,
    layout: TargetDataLayout,
    leaves: &mut Vec<(u64, u64)>,
) -> Option<()> {
    match ty {
        IRType::Struct { fields, packed } => {
            for (index, field) in fields.iter().enumerate() {
                let offset = struct_field_byte_offset(fields, *packed, index, layout)?;
                push_leaves(&field.ty, base.checked_add(offset)?, layout, leaves)?;
            }
        }
        IRType::Array { element_type, size } => {
            let element = sizeof_type(element_type, layout)?;
            if element == 0 {
                return Some(());
            }
            for index in 0..*size {
                let offset = index.checked_mul(element)?;
                push_leaves(element_type, base.checked_add(offset)?, layout, leaves)?;
                if leaves.len() > MAX_COPY_PIECES {
                    return None;
                }
            }
        }
        other => {
            let size = sizeof_type(other, layout)?;
            if size > 0 {
                leaves.push((base, size));
            }
        }
    }
    (leaves.len() <= MAX_COPY_PIECES).then_some(())
}

fn integer_of_size(size: u64) -> ScalarType {
    match size {
        1 => ScalarType::I8,
        2 => ScalarType::I16,
        4 => ScalarType::I32,
        _ => ScalarType::I64,
    }
}

/// Splits an aggregate into the scalar pieces a copy moves, each as an integer of at
/// most a pointer's width at the offset its bytes sit at. A float or a pointer moves
/// as its bits, so the copy needs no float register.
fn copy_pieces(
    ty: &IRType<'_>,
    layout: TargetDataLayout,
) -> Result<Vec<(u64, ScalarType)>, FromIRError> {
    let mut leaves = Vec::new();
    push_leaves(ty, 0, layout, &mut leaves).ok_or(FromIRError::UnsupportedType)?;
    let word = u64::try_from(MirType::Scalar(ScalarType::Ptr).size_bytes(layout.pointer_width))
        .map_err(|_| FromIRError::UnsupportedType)?;
    let mut pieces = Vec::new();
    for (offset, size) in leaves {
        let mut done = 0;
        while done < size {
            let remaining = size - done;
            let chunk = [8u64, 4, 2, 1]
                .into_iter()
                .find(|&chunk| chunk <= remaining && chunk <= word)
                .ok_or(FromIRError::UnsupportedType)?;
            pieces.push((offset + done, integer_of_size(chunk)));
            done += chunk;
        }
    }
    if pieces.len() > MAX_COPY_PIECES {
        return Err(FromIRError::UnsupportedType);
    }
    Ok(pieces)
}

/// The address of the piece at `offset` from `base`. A backend takes a memory access
/// only at displacement zero, so a non-zero offset is first added into a register.
fn piece_address(
    base: &Register,
    offset: u64,
    vreg_alloc: &mut VirtualRegAllocator,
    instructions: &mut Vec<Instruction>,
) -> Result<AddressMode, FromIRError> {
    let base = if offset == 0 {
        base.clone()
    } else {
        let wide = i64::try_from(offset).map_err(|_| FromIRError::UnsupportedType)?;
        let address = Register::Virtual(vreg_alloc.allocate_gpr());
        instructions.extend(displace_pointer(
            address.clone(),
            base.clone(),
            wide,
            vreg_alloc,
        ));
        address
    };
    Ok(AddressMode::BaseOffset {
        base,
        offset: 0,
        address_space: Default::default(),
    })
}

/// Alignment an access at `offset` from an address of alignment `base` can state.
fn piece_alignment(base: u8, offset: u64) -> u8 {
    let from_offset = if offset == 0 {
        u8::MAX
    } else {
        1u8 << offset.trailing_zeros().min(7)
    };
    base.min(from_offset)
}

fn piece_attrs(
    base: u8,
    offset: u64,
    scalar: ScalarType,
    volatile: bool,
    layout: TargetDataLayout,
) -> Result<MemoryAttrs, FromIRError> {
    Ok(MemoryAttrs {
        align: access_alignment(
            None,
            MirType::Scalar(scalar),
            piece_alignment(base, offset),
            layout,
            "aggregate copy",
        )?,
        volatile,
        stated_align: false,
    })
}

fn frame_slot(
    size: u64,
    align: u64,
    vreg_alloc: &mut VirtualRegAllocator,
    reservations: &mut HashMap<VirtualReg, usize>,
    alignments: &mut HashMap<VirtualReg, usize>,
) -> Result<VirtualReg, FromIRError> {
    if align > 16 {
        return Err(FromIRError::UnsupportedAlignment(format!(
            "an aggregate of {size} bytes needs align {align}: frames are only 16-byte aligned and frame realignment is not implemented yet"
        )));
    }
    let slot = vreg_alloc.allocate_gpr();
    let slots_needed =
        usize::try_from(size.div_ceil(8)).map_err(|_| FromIRError::UnsupportedType)?;
    if slots_needed > 1 {
        reservations.insert(slot, slots_needed - 1);
    }
    if align > 8 {
        alignments.insert(slot, align as usize);
    }
    Ok(slot)
}

/// Lowers a whole-aggregate `load`: `dst` becomes the address of a fresh frame copy of
/// the bytes at `base`. `ty` is resolved.
#[allow(clippy::too_many_arguments)]
pub(super) fn load(
    dst: Register,
    base: Register,
    base_align: u8,
    ty: &IRType<'_>,
    volatile: bool,
    layout: TargetDataLayout,
    vreg_alloc: &mut VirtualRegAllocator,
    reservations: &mut HashMap<VirtualReg, usize>,
    alignments: &mut HashMap<VirtualReg, usize>,
) -> Result<Vec<Instruction>, FromIRError> {
    let size = sizeof_type(ty, layout).ok_or(FromIRError::UnsupportedType)?;
    let slot = frame_slot(
        size,
        align_of_type(ty, layout),
        vreg_alloc,
        reservations,
        alignments,
    )?;
    let mut instructions = vec![Instruction::StackAddr {
        dst: dst.clone(),
        slot,
    }];
    for (offset, scalar) in copy_pieces(ty, layout)? {
        let attrs = piece_attrs(base_align, offset, scalar, volatile, layout)?;
        let from = piece_address(&base, offset, vreg_alloc, &mut instructions)?;
        let to = piece_address(&dst, offset, vreg_alloc, &mut instructions)?;
        let temp = Register::Virtual(vreg_alloc.allocate_gpr());
        instructions.push(Instruction::Load {
            ty: MirType::Scalar(scalar),
            dst: temp.clone(),
            addr: from,
            attrs,
        });
        instructions.push(Instruction::Store {
            ty: MirType::Scalar(scalar),
            src: Operand::Register(temp),
            addr: to,
            attrs,
        });
    }
    Ok(instructions)
}

/// Lowers a whole-aggregate `store` of the aggregate at address `src` to `base`.
/// `ty` is resolved.
#[allow(clippy::too_many_arguments)]
pub(super) fn store(
    base: Register,
    base_align: u8,
    src: Register,
    ty: &IRType<'_>,
    volatile: bool,
    layout: TargetDataLayout,
    vreg_alloc: &mut VirtualRegAllocator,
) -> Result<Vec<Instruction>, FromIRError> {
    let mut instructions = Vec::new();
    for (offset, scalar) in copy_pieces(ty, layout)? {
        let attrs = piece_attrs(base_align, offset, scalar, volatile, layout)?;
        let from = piece_address(&src, offset, vreg_alloc, &mut instructions)?;
        let to = piece_address(&base, offset, vreg_alloc, &mut instructions)?;
        let temp = Register::Virtual(vreg_alloc.allocate_gpr());
        instructions.push(Instruction::Load {
            ty: MirType::Scalar(scalar),
            dst: temp.clone(),
            addr: from,
            attrs,
        });
        instructions.push(Instruction::Store {
            ty: MirType::Scalar(scalar),
            src: Operand::Register(temp),
            addr: to,
            attrs,
        });
    }
    Ok(instructions)
}
