//! One x86_64 call site. A call that names its callee and a call through a pointer
//! place their arguments the same way; only the widening a prototype asks for, the
//! vector count a variadic callee reads, and the call instruction itself differ.

use super::function::{
    AbiExtensions, ArgDesc, ArgLocation, WordPlace, classify_argument_descs_with_sret,
    passed_by_reference,
};
use super::instruction::{
    StagedWord, copy_exact_bytes_to_stack, duplicate_win64_float_arguments, load_aggregate_word,
    load_register_arguments, widen_external_argument,
};
use super::*;
use crate::mir_codegen::common::align_up;

/// What a call site needs beyond its arguments: the callee's name when it has one,
/// the pointer it calls through when it does not, the frame slot a returned
/// aggregate is written to, and whether the callee reads a vector count.
pub(super) struct CallSite<'a> {
    pub(super) name: Option<&'a str>,
    pub(super) target: Option<&'a Operand>,
    pub(super) descs: &'a [ArgDesc],
    pub(super) args: &'a [Operand],
    pub(super) sret_result: Option<i32>,
    pub(super) variadic: bool,
}

/// Places the arguments, calls, and unwinds the outgoing stack.
pub(super) fn emit<W: Write, RA: LocalRegisterAllocator<PhysReg = &'static str>>(
    site: &CallSite<'_>,
    writer: &mut RaxTrackingWriter<W>,
    reg_alloc: &RA,
    stack_slots: &HashMap<VirtualReg, i32>,
    target_os: TargetOperatingSystem,
    abi_extensions: &AbiExtensions,
) -> Result<(), LaminaError> {
    let abi = X86ABI::new(target_os);
    let arg_regs = abi.arg_registers();
    let types: Vec<MirType> = site.descs.iter().map(|desc| desc.ty).collect();
    let locations =
        classify_argument_descs_with_sret(site.descs, target_os, site.sret_result.is_some());
    // One entry per eight-byte slot, lowest slot first: an aggregate on the
    // stack occupies as many as it has words.
    // One entry per stack slot, lowest first. `None` is a padding slot that the
    // classifier left between a 16-byte-aligned aggregate and the one before.
    let mut stack_args: Vec<Option<(usize, usize)>> = Vec::new();
    for (index, location) in locations.iter().enumerate() {
        match location {
            ArgLocation::Stack(_) => stack_args.push(Some((index, 0))),
            ArgLocation::StackWords { first, words } => {
                while stack_args.len() < *first {
                    stack_args.push(None);
                }
                stack_args.extend((0..*words).map(|word| Some((index, word))));
            }
            ArgLocation::Gpr(_)
            | ArgLocation::Xmm(_)
            | ArgLocation::GprWords { .. }
            | ArgLocation::EightbyteWords { .. } => {}
        }
    }
    let num_stack_args = stack_args.len();

    // Windows takes an aggregate that is not register sized as a pointer to a
    // copy belonging to the caller, which goes above the outgoing arguments.
    let mut copy_offsets: Vec<Option<usize>> = vec![None; site.descs.len()];
    let mut copy_bytes = 0;
    if target_os == TargetOperatingSystem::Windows {
        let base = align_up(
            windows::SHADOW_SPACE_SIZE as usize + (num_stack_args * stack::SLOT_SIZE),
            stack::ALIGNMENT,
        );
        for (index, desc) in site.descs.iter().enumerate() {
            if let Some(aggregate) = desc.aggregate
                && passed_by_reference(aggregate, target_os)
            {
                copy_offsets[index] = Some(base + copy_bytes);
                copy_bytes += align_up(aggregate.words() * stack::SLOT_SIZE, stack::ALIGNMENT);
            }
        }
    }

    if target_os == TargetOperatingSystem::Windows {
        // Every Windows call gets home space, argument count aside:
        // the callee may spill into it whether or not it was passed
        // register arguments.
        let shadow_space = windows::SHADOW_SPACE_SIZE as usize;
        // Rounded up so rsp stays 16-aligned at the call; the
        // padding lands above the arguments, leaving their offsets
        // from rsp unchanged.
        let total_stack = align_up(
            shadow_space + (num_stack_args * stack::SLOT_SIZE),
            stack::ALIGNMENT,
        ) + copy_bytes;
        if total_stack > 0 {
            writeln!(writer, "    subq ${total_stack}, %rsp")?;
        }
        for (index, at) in copy_offsets.iter().enumerate() {
            let Some(at) = *at else {
                continue;
            };
            let size = site.descs[index].aggregate.map_or(0, |a| a.size);
            load_operand_to_rax(&site.args[index], writer, reg_alloc, stack_slots)?;
            copy_exact_bytes_to_stack(writer, size, "rax", at)?;
            writer.rax_value = None;
        }
        for (slot, entry) in stack_args.iter().enumerate() {
            let stack_offset = shadow_space + (slot * stack::SLOT_SIZE);
            // Windows never pads: its aggregates that need a stack slot are not
            // 16-byte aligned by value.
            let Some((index, word)) = *entry else {
                return Err(LaminaError::ValidationError(
                    "Windows stack arguments cannot need alignment padding".to_string(),
                ));
            };
            if let Some(at) = copy_offsets[index] {
                writeln!(writer, "    leaq {at}(%rsp), %rax")?;
                writeln!(writer, "    movq %rax, {stack_offset}(%rsp)")?;
                continue;
            }
            if site.descs[index].aggregate.is_some() {
                load_aggregate_word(&site.args[index], word, writer, reg_alloc, stack_slots)?;
            } else {
                load_operand_to_rax(&site.args[index], writer, reg_alloc, stack_slots)?;
                if let Some(name) = site.name {
                    widen_external_argument(writer, abi_extensions, name, index, "rax")?;
                }
            }
            writeln!(writer, "    movq %rax, {stack_offset}(%rsp)")?;
        }
    } else {
        // An odd number of pushes would leave rsp 8 off at the call.
        if num_stack_args % 2 == 1 {
            writeln!(writer, "    subq $8, %rsp")?;
        }
        for entry in stack_args.iter().rev() {
            let Some((index, word)) = *entry else {
                writeln!(writer, "    pushq $0")?;
                continue;
            };
            if site.descs[index].aggregate.is_some() {
                load_aggregate_word(&site.args[index], word, writer, reg_alloc, stack_slots)?;
            } else {
                load_operand_to_rax(&site.args[index], writer, reg_alloc, stack_slots)?;
                if let Some(name) = site.name {
                    widen_external_argument(writer, abi_extensions, name, index, "rax")?;
                }
            }
            writeln!(writer, "    pushq %rax")?;
        }
    }

    // A by-reference aggregate's register holds the address of the copy, so
    // the scalar pass must leave it alone.
    let mut scalar_locations = locations.clone();
    for (index, at) in copy_offsets.iter().enumerate() {
        if at.is_some() {
            scalar_locations[index] = ArgLocation::Stack(usize::MAX);
        }
    }

    // Aggregate words are staged on the stack so that loading the scalar
    // arguments cannot clobber an argument register already filled.
    let mut staged: Vec<StagedWord> = Vec::new();
    for (index, location) in locations.iter().enumerate() {
        if copy_offsets[index].is_some() {
            continue;
        }
        if let ArgLocation::EightbyteWords { places, words } = location {
            for (word, place) in places[..*words].iter().enumerate() {
                load_aggregate_word(&site.args[index], word, writer, reg_alloc, stack_slots)?;
                writeln!(writer, "    pushq %rax")?;
                staged.push(match place {
                    WordPlace::Gpr(gpr) => StagedWord::Gpr(arg_regs[*gpr]),
                    WordPlace::Xmm(xmm) => StagedWord::Xmm(*xmm),
                });
            }
            continue;
        }
        let (first, words) = match location {
            ArgLocation::GprWords { first, words } => (*first, *words),
            ArgLocation::Gpr(gpr) if site.descs[index].aggregate.is_some() => (*gpr, 1),
            _ => continue,
        };
        for word in 0..words {
            load_aggregate_word(&site.args[index], word, writer, reg_alloc, stack_slots)?;
            writeln!(writer, "    pushq %rax")?;
            staged.push(StagedWord::Gpr(arg_regs[first + word]));
        }
    }

    load_register_arguments(
        writer,
        site.args,
        &types,
        &scalar_locations,
        arg_regs,
        reg_alloc,
        stack_slots,
    )?;

    for word in staged.iter().rev() {
        match word {
            StagedWord::Gpr(register) => writeln!(writer, "    popq %{register}")?,
            StagedWord::Xmm(xmm) => {
                writeln!(writer, "    popq %r11")?;
                writeln!(writer, "    movq %r11, %xmm{xmm}")?;
            }
        }
    }
    for (index, location) in locations.iter().enumerate() {
        if let ArgLocation::Gpr(gpr) = location
            && site.descs[index].aggregate.is_none()
        {
            if let Some(name) = site.name {
                widen_external_argument(writer, abi_extensions, name, index, arg_regs[*gpr])?;
            }
        }
    }
    if target_os == TargetOperatingSystem::Windows {
        duplicate_win64_float_arguments(writer, &locations, arg_regs)?;
    }

    for (index, at) in copy_offsets.iter().enumerate() {
        if let (Some(at), ArgLocation::Gpr(gpr)) = (*at, &locations[index]) {
            writeln!(writer, "    leaq {at}(%rsp), %{}", arg_regs[*gpr])?;
        }
    }
    // r11 is caller-saved and never an argument register, so the target
    // survives the sret pointer and the vector count.
    if let Some(target) = site.target {
        load_operand_to_rax(target, writer, reg_alloc, stack_slots)?;
        writeln!(writer, "    movq %rax, %r11")?;
    }
    if let Some(offset) = site.sret_result {
        writeln!(writer, "    leaq {offset}(%rbp), %{}", arg_regs[0])?;
    }
    if target_os != TargetOperatingSystem::Windows && site.variadic {
        let xmm_count: usize = locations.iter().map(ArgLocation::xmm_words).sum();
        writeln!(writer, "    movl ${xmm_count}, %eax")?;
    }
    match site.name {
        Some(name) => {
            let mangled_name = abi.mangle_function_name(name);
            writeln!(writer, "    call {mangled_name}")?;
        }
        None => writeln!(writer, "    call *%r11")?,
    }

    if target_os == TargetOperatingSystem::Windows {
        // Every Windows call gets home space, argument count aside:
        // the callee may spill into it whether or not it was passed
        // register arguments.
        let shadow_space = windows::SHADOW_SPACE_SIZE as usize;
        let total_stack = align_up(
            shadow_space + (num_stack_args * stack::SLOT_SIZE),
            stack::ALIGNMENT,
        ) + copy_bytes;
        if total_stack > 0 {
            writeln!(writer, "    addq ${total_stack}, %rsp")?;
        }
    } else if num_stack_args > 0 {
        let pushed = num_stack_args * stack::SLOT_SIZE;
        let padded = align_up(pushed, stack::ALIGNMENT);
        writeln!(writer, "    addq ${padded}, %rsp")?;
    }

    Ok(())
}
