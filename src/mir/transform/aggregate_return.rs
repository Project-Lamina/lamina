//! Gives a function that returns an aggregate from several stack slots one return slot.
//!
//! Each return path copies the fields it wrote into a single slot and returns that. The
//! copies are typed by the accesses the function makes to the returned slots, so the
//! result still splits into scalars, and a caller that inlines the function sees one
//! definition of the returned pointer.

use crate::mir::{
    AddressMode, AddressSpace, Function, Instruction, MirType, Operand, Register, ScalarType,
    VirtualReg, transform::loop_rotation::next_virtual_id,
};
use lamina_mir::{Located, MemoryAttrs, RegisterClass};
use std::collections::{BTreeMap, HashMap};

type Pieces = BTreeMap<i64, (MirType, MemoryAttrs)>;

/// Unifies the return slots of `func` when it returns an aggregate from more than one of
/// them. Returns true when the function changed.
pub(crate) fn unify_returns(func: &mut Function) -> bool {
    if func.sig.ret_aggregate.is_none() {
        return false;
    }
    let mut rets = Vec::new();
    for (block_index, block) in func.blocks.iter().enumerate() {
        for (index, inst) in block.instructions.iter().enumerate() {
            if let Instruction::Ret {
                value: Some(Operand::Register(reg)),
            } = &inst.instr
            {
                rets.push((block_index, index, reg.clone()));
            }
        }
    }
    if rets.len() < 2 {
        return false;
    }
    let Some(derived) = derive(func) else {
        return false;
    };
    let mut slots = Vec::new();
    for (_, _, reg) in &rets {
        match derived.get(reg) {
            Some(&(slot, 0)) => slots.push(slot),
            _ => return false,
        }
    }
    slots.sort_unstable_by_key(|slot| (slot.id, slot.class));
    slots.dedup();
    if slots.len() < 2 {
        return false;
    }
    let mut pieces = Pieces::new();
    for slot in &slots {
        if collect_pieces(func, *slot, &derived, &mut pieces).is_none() {
            return false;
        }
    }
    if !disjoint(&pieces) {
        return false;
    }
    let words = slots
        .iter()
        .map(|slot| func.stack_reservations.get(slot).copied().unwrap_or(0))
        .max()
        .unwrap_or(0);

    let mut next_id = fresh_base(func);
    let mut fresh = |class: RegisterClass| {
        let reg = VirtualReg::new(next_id, class);
        next_id += 1;
        reg
    };
    let unified_slot = fresh(RegisterClass::Gpr);
    let unified_ptr = fresh(RegisterClass::Gpr);
    let unified_label = unique_label(func, "ret_unified");
    let entry = func
        .blocks
        .iter()
        .position(|block| block.label == func.entry)
        .unwrap_or(0);
    func.blocks[entry].instructions.insert(
        0,
        Instruction::StackAddr {
            dst: Register::Virtual(unified_ptr),
            slot: unified_slot,
        }
        .into(),
    );
    func.stack_reservations.insert(unified_slot, words);

    let mut by_block: BTreeMap<usize, Vec<(usize, Register)>> = BTreeMap::new();
    for (block_index, index, reg) in rets {
        by_block.entry(block_index).or_default().push((index, reg));
    }
    for (block_index, sites) in by_block {
        let block = &mut func.blocks[block_index];
        for (index, reg) in sites.into_iter().rev() {
            let mut replacement: Vec<Located> = Vec::new();
            for (&offset, (ty, attrs)) in &pieces {
                let class = match ty {
                    MirType::Scalar(ScalarType::F32 | ScalarType::F64) => RegisterClass::Fpr,
                    _ => RegisterClass::Gpr,
                };
                let tmp = Register::Virtual(fresh(class));
                let offset = i16::try_from(offset).unwrap_or(i16::MAX);
                replacement.push(
                    Instruction::Load {
                        ty: *ty,
                        dst: tmp.clone(),
                        addr: at(reg.clone(), offset),
                        attrs: *attrs,
                    }
                    .into(),
                );
                replacement.push(
                    Instruction::Store {
                        ty: *ty,
                        src: Operand::Register(tmp),
                        addr: at(Register::Virtual(unified_ptr), offset),
                        attrs: *attrs,
                    }
                    .into(),
                );
            }
            replacement.push(
                Instruction::Jmp {
                    target: unified_label.clone(),
                }
                .into(),
            );
            block.instructions.splice(index..=index, replacement);
        }
    }
    func.blocks.push(crate::mir::Block {
        label: unified_label,
        instructions: vec![
            Instruction::Ret {
                value: Some(Operand::Register(Register::Virtual(unified_ptr))),
            }
            .into(),
        ],
    });
    true
}

fn at(base: Register, offset: i16) -> AddressMode {
    AddressMode::BaseOffset {
        address_space: AddressSpace::default(),
        base,
        offset,
    }
}

fn fresh_base(func: &Function) -> u32 {
    let slot_ids = func
        .stack_reservations
        .keys()
        .chain(func.stack_alignments.keys())
        .map(|slot| slot.id)
        .chain(
            func.blocks
                .iter()
                .flat_map(|block| &block.instructions)
                .filter_map(|inst| match &inst.instr {
                    Instruction::StackAddr { slot, .. } => Some(slot.id),
                    _ => None,
                }),
        );
    slot_ids
        .max()
        .map_or(0, |id| id + 1)
        .max(next_virtual_id(func))
}

fn unique_label(func: &Function, base: &str) -> String {
    let mut label = base.to_string();
    while func.blocks.iter().any(|block| block.label == label) {
        label.push('_');
    }
    label
}

/// Each register that holds a pointer into one stack slot, with the byte offset it points
/// at. `None` when such a register is also written by something else.
fn derive(func: &Function) -> Option<HashMap<Register, (VirtualReg, i64)>> {
    let mut defs: HashMap<Register, usize> = HashMap::new();
    for inst in func.blocks.iter().flat_map(|block| &block.instructions) {
        if let Some(dst) = inst.instr.def_reg() {
            *defs.entry(dst.clone()).or_default() += 1;
        }
    }
    let mut derived: HashMap<Register, (VirtualReg, i64)> = HashMap::new();
    for inst in func.blocks.iter().flat_map(|block| &block.instructions) {
        if let Instruction::StackAddr { dst, slot } = &inst.instr {
            derived.insert(dst.clone(), (*slot, 0));
        }
    }
    loop {
        let mut grew = false;
        for inst in func.blocks.iter().flat_map(|block| &block.instructions) {
            let (dst, base, step) = match &inst.instr {
                Instruction::Lea { dst, base, offset } => (dst, base, i64::from(*offset)),
                Instruction::Copy {
                    ty: MirType::Scalar(ScalarType::Ptr),
                    dst,
                    src: Operand::Register(base),
                } => (dst, base, 0),
                _ => continue,
            };
            if derived.contains_key(dst) {
                continue;
            }
            if let Some(&(slot, offset)) = derived.get(base) {
                derived.insert(dst.clone(), (slot, offset + step));
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    derived
        .keys()
        .all(|reg| defs.get(reg).copied().unwrap_or(0) == 1)
        .then_some(derived)
}

/// Adds the typed accesses the function makes to `slot`. `None` when the slot is used
/// in any other way, or an access is not a plain scalar.
fn collect_pieces(
    func: &Function,
    slot: VirtualReg,
    derived: &HashMap<Register, (VirtualReg, i64)>,
    pieces: &mut Pieces,
) -> Option<()> {
    for inst in func.blocks.iter().flat_map(|block| &block.instructions) {
        let touches = inst
            .instr
            .use_regs()
            .into_iter()
            .any(|reg| derived.get(reg).is_some_and(|&(s, _)| s == slot));
        if !touches {
            continue;
        }
        match &inst.instr {
            Instruction::Lea { .. } => {}
            Instruction::Copy {
                ty: MirType::Scalar(ScalarType::Ptr),
                ..
            } => {}
            Instruction::Ret { .. } => {}
            Instruction::Load {
                ty: ty @ MirType::Scalar(_),
                addr:
                    AddressMode::BaseOffset {
                        base,
                        offset,
                        address_space,
                    },
                attrs,
                ..
            } if *address_space == AddressSpace::default() && !attrs.volatile => {
                let &(s, base_offset) = derived.get(base)?;
                if s != slot {
                    return None;
                }
                add_piece(pieces, base_offset + i64::from(*offset), *ty, *attrs)?;
            }
            Instruction::Store {
                ty: ty @ MirType::Scalar(_),
                src,
                addr:
                    AddressMode::BaseOffset {
                        base,
                        offset,
                        address_space,
                    },
                attrs,
            } if *address_space == AddressSpace::default()
                && !attrs.volatile
                && !matches!(src, Operand::Register(r) if derived.contains_key(r)) =>
            {
                let &(s, base_offset) = derived.get(base)?;
                if s != slot {
                    return None;
                }
                add_piece(pieces, base_offset + i64::from(*offset), *ty, *attrs)?;
            }
            _ => return None,
        }
    }
    Some(())
}

fn add_piece(pieces: &mut Pieces, offset: i64, ty: MirType, attrs: MemoryAttrs) -> Option<()> {
    if !(0..=i64::from(i16::MAX)).contains(&offset) {
        return None;
    }
    match pieces.get(&offset).copied() {
        Some((existing, attrs)) if existing != ty => {
            let width = scalar_size(existing).max(scalar_size(ty));
            pieces.insert(offset, (int_of(width), attrs));
            Some(())
        }
        Some(_) => Some(()),
        None => {
            pieces.insert(offset, (ty, attrs));
            Some(())
        }
    }
}

fn int_of(size: i64) -> MirType {
    MirType::Scalar(match size {
        1 => ScalarType::I8,
        2 => ScalarType::I16,
        4 => ScalarType::I32,
        _ => ScalarType::I64,
    })
}

fn scalar_size(ty: MirType) -> i64 {
    match ty {
        MirType::Scalar(ScalarType::I1 | ScalarType::I8) => 1,
        MirType::Scalar(ScalarType::I16) => 2,
        MirType::Scalar(ScalarType::I32 | ScalarType::F32) => 4,
        _ => 8,
    }
}

/// Whether no two pieces share a byte.
fn disjoint(pieces: &Pieces) -> bool {
    let mut end = i64::MIN;
    for (&offset, (ty, _)) in pieces {
        if offset < end {
            return false;
        }
        end = offset + scalar_size(*ty);
    }
    true
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};

    fn lowered(source: &str) -> Function {
        let ir = parse_module(source).expect("parse");
        let module = from_ir_x86_64(&ir, "aggregate return").expect("lower");
        module.functions.get("pick").expect("pick").clone()
    }

    fn returns(func: &Function) -> usize {
        func.blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .filter(|inst| matches!(inst.instr, Instruction::Ret { .. }))
            .count()
    }

    const TWO_SLOTS: &str = "type !Pair = struct { a: i64, b: i64 }\nfn @pick(i64 %c) -> !Pair {\nentry:\n    %x = stackalloc.!Pair\n    %y = stackalloc.!Pair\n    %fx = getfieldptr.!Pair %x, 0\n    store.i64 %fx, %c\n    %fy = getfieldptr.!Pair %y, 1\n    store.i64 %fy, 2\n    %t = eq.i64 %c, 0\n    br %t, left, right\nleft:\n    %v = load.!Pair %x\n    ret.!Pair %v\nright:\n    %w = load.!Pair %y\n    ret.!Pair %w\n}\n";

    #[test]
    fn two_return_slots_become_one() {
        let mut func = lowered(TWO_SLOTS);
        assert_eq!(returns(&func), 2);
        assert!(unify_returns(&mut func));
        assert_eq!(returns(&func), 1);
    }

    #[test]
    fn a_returned_slot_passed_to_a_call_keeps_its_returns() {
        let mut func = lowered(
            "type !Pair = struct { a: i64, b: i64 }\nfn @sink(!Pair %p) -> void {\nentry:\n    ret.void\n}\nfn @pick(i64 %c) -> !Pair {\nentry:\n    %x = stackalloc.!Pair\n    %y = stackalloc.!Pair\n    %t = eq.i64 %c, 0\n    br %t, left, right\nleft:\n    %v = load.!Pair %x\n    call @sink(%v)\n    ret.!Pair %v\nright:\n    %w = load.!Pair %y\n    ret.!Pair %w\n}\n",
        );
        assert!(!unify_returns(&mut func));
        assert_eq!(returns(&func), 2);
    }

    #[test]
    fn a_single_return_slot_is_left_alone() {
        let mut func = lowered(
            "type !Pair = struct { a: i64, b: i64 }\nfn @pick(i64 %c) -> !Pair {\nentry:\n    %x = stackalloc.!Pair\n    %fx = getfieldptr.!Pair %x, 0\n    store.i64 %fx, %c\n    %v = load.!Pair %x\n    ret.!Pair %v\n}\n",
        );
        assert!(!unify_returns(&mut func));
    }

    #[test]
    fn a_caller_inlines_a_callee_with_two_return_slots() {
        use crate::mir::transform::ModuleInlining;
        let source = "type !Pair = struct { a: i64, b: i64 }\nfn @pick(i64 %c) -> !Pair {\nentry:\n    %x = stackalloc.!Pair\n    %y = stackalloc.!Pair\n    %fx = getfieldptr.!Pair %x, 0\n    store.i64 %fx, %c\n    %fy = getfieldptr.!Pair %y, 1\n    store.i64 %fy, 2\n    %t = eq.i64 %c, 0\n    br %t, left, right\nleft:\n    %v = load.!Pair %x\n    ret.!Pair %v\nright:\n    %w = load.!Pair %y\n    ret.!Pair %w\n}\nfn @main(i64 %c) -> i64 {\nentry:\n    %p = call @pick(%c)\n    %s = stackalloc.!Pair\n    store.!Pair %s, %p\n    %f = getfieldptr.!Pair %s, 0\n    %x = load.i64 %f\n    ret.i64 %x\n}\n";
        let ir = parse_module(source).expect("parse");
        let mut module = from_ir_x86_64(&ir, "aggregate inline").expect("lower");
        let count = ModuleInlining::new()
            .inline_functions(&mut module)
            .expect("inline");
        assert_eq!(count, 1);
        let main = module.get_function("main").expect("main");
        assert!(
            !main
                .blocks
                .iter()
                .flat_map(|block| &block.instructions)
                .any(|inst| matches!(inst.instr, Instruction::Call { .. })),
        );
    }
}
