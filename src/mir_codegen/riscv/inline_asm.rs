//! Inline assembly for RISC-V.
//!
//! Operands are bound to physical registers around the template: inputs are loaded from
//! their frame slots into the bound register, the template is emitted with `$n` replaced
//! by the register of operand `n`, and the output register is written back to the result
//! slot. Nothing the emitter produces keeps a value in a register across an instruction,
//! so a clobber costs nothing and needs no reporting; the prologue already saves the
//! registers the ABI asks a callee to keep. The registers the body itself depends on —
//! the stack pointer, the frame pointer, the thread and global pointers and the zero
//! register — are refused instead.

use super::util::{load_operand_to_register, store_register_to_register};
use super::{RiscVRegAlloc, RiscVTarget, extend_integer, target_pointer};
use crate::mir_codegen::inline_asm::{
    Binding, GPR_CLASS, RegisterFacts, expand_template, output_register, result_is_bool, spell_bare,
};
use crate::{
    error::LaminaError,
    mir::{AsmOperand, Instruction as MirInst, Register, VirtualReg},
};
use std::collections::HashMap;
use std::io::Write;

/// Every general register, under the name the ABI gives it and the number it carries.
const GPRS: [&str; 64] = [
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "fp", "s1", "a0", "a1", "a2", "a3",
    "a4", "a5", "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3",
    "t4", "t5", "t6", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11",
    "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24",
    "x25", "x26", "x27", "x28", "x29", "x30",
];

const FPRS: [&str; 32] = [
    "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "fs0", "fs1", "fa0", "fa1", "fa2",
    "fa3", "fa4", "fa5", "fa6", "fa7", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7", "fs8", "fs9",
    "fs10", "fs11", "ft8", "ft9", "ft10", "ft11",
];

/// Registers the emitted body depends on between instructions, under every name they
/// answer to: the stack and frame pointers, the global and thread pointers, and the
/// register that reads as zero.
const RESERVED: [&str; 10] = ["sp", "x2", "s0", "fp", "x8", "gp", "x3", "tp", "x4", "zero"];

/// Registers a `reg` operand may be given, in the order they are tried. All are
/// caller-saved, so the body owes nothing to the function around it.
const REG_CLASS_POOL: [&str; 7] = ["a0", "a1", "a2", "a3", "a4", "a5", "a6"];

const FACTS: RegisterFacts = RegisterFacts {
    operand: named_register,
    clobber: clobber_register,
    spell: spell_bare,
    pool: &REG_CLASS_POOL,
};

/// The register the result leaves in, which is also the one a value is staged in.
const RESULT: &str = "t0";

fn is_fpr(reg: &str) -> bool {
    FPRS.contains(&reg)
}

fn unknown_register(name: &str) -> LaminaError {
    LaminaError::ValidationError(format!(
        "unknown register '{name}' in inline asm for RISC-V; valid names are: {GPR_CLASS}, \
         the general registers under their ABI or x names, and {}",
        FPRS.join(", ")
    ))
}

/// Resolves a register name written in an operand or a clobber to its canonical spelling.
fn named_register(name: &str) -> Result<&'static str, LaminaError> {
    if RESERVED.contains(&name) {
        return Err(LaminaError::ValidationError(format!(
            "register '{name}' is reserved by the RISC-V backend and cannot be named by \
             inline asm; sp is the stack pointer, s0 the frame pointer, gp and tp the \
             global and thread pointers, and zero reads as zero"
        )));
    }
    GPRS.iter()
        .chain(FPRS.iter())
        .copied()
        .find(|candidate| *candidate == name)
        .ok_or_else(|| unknown_register(name))
}

fn clobber_register(name: &str) -> Result<Option<&'static str>, LaminaError> {
    named_register(name).map(Some)
}

pub(super) fn bind(
    operands: &[AsmOperand],
    clobbers: &[String],
) -> Result<Vec<Binding>, LaminaError> {
    FACTS.bind(operands, clobbers)
}

pub(super) fn emit<W: Write>(
    inst: &MirInst,
    writer: &mut W,
    reg_alloc: &RiscVRegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    target: RiscVTarget,
) -> Result<(), LaminaError> {
    let MirInst::InlineAsm {
        template,
        operands,
        clobbers,
        result,
        ..
    } = inst
    else {
        return Err(LaminaError::ValidationError(
            "RISC-V inline assembly emitter received another instruction".to_string(),
        ));
    };
    let bindings = bind(operands, clobbers)?;
    let text = expand_template(template, &bindings)?;

    for (operand, binding) in operands.iter().zip(&bindings) {
        if let AsmOperand::Input { value, .. } = operand {
            if is_fpr(binding.phys) {
                return Err(LaminaError::ValidationError(format!(
                    "inline asm input is bound to the floating-point register \
                     '{}'; a RISC-V asm operand must take a general register",
                    binding.phys
                )));
            }
            load_operand_to_register(value, writer, reg_alloc, stack_slots, binding.phys)?;
        }
    }

    for line in text.lines() {
        let line = line.trim();
        if !line.is_empty() {
            writeln!(writer, "    {line}")?;
        }
    }

    let Some((dst, ty)) = result else {
        return Ok(());
    };
    let output = output_register(operands, &bindings)?;
    if is_fpr(output) {
        return Err(LaminaError::ValidationError(format!(
            "inline asm result of type {ty} is read from the floating-point register \
             '{output}'; a RISC-V asm result must come from a general register"
        )));
    }
    if output != RESULT {
        writeln!(writer, "    mv {RESULT}, {output}")?;
    }
    extend_integer(
        writer,
        RESULT,
        ty,
        target,
        !result_is_bool(ty),
        target_pointer(target),
    )?;
    match dst {
        Register::Virtual(vreg) => {
            store_register_to_register(RESULT, vreg, writer, reg_alloc, stack_slots)?;
            Ok(())
        }
        Register::Physical(physical) => {
            if physical.name != RESULT {
                writeln!(writer, "    mv {}, {RESULT}", physical.name)?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{Operand, VirtualReg};

    fn out(location: &str) -> AsmOperand {
        AsmOperand::Output {
            location: location.to_owned(),
        }
    }

    fn input(location: &str, vreg: u32) -> AsmOperand {
        AsmOperand::Input {
            location: location.to_owned(),
            value: Operand::Register(Register::Virtual(VirtualReg::gpr(vreg))),
        }
    }

    fn spellings(operands: &[AsmOperand], clobbers: &[String]) -> Vec<String> {
        bind(operands, clobbers)
            .expect("bind")
            .into_iter()
            .map(|binding| binding.spelled)
            .collect()
    }

    #[test]
    fn riscv_named_operands_keep_their_registers() {
        assert_eq!(
            spellings(&[input("a3", 0), out("a4")], &[]),
            vec!["a3".to_owned(), "a4".to_owned()]
        );
    }

    #[test]
    fn riscv_reg_class_operands_avoid_named_operands_and_clobbers() {
        let clobbers = vec!["a1".to_owned(), "memory".to_owned()];
        let spelled = spellings(&[input("a0", 0), out(GPR_CLASS)], &clobbers);
        assert_eq!(spelled[0], "a0");
        assert!(
            spelled[1] != "a0" && spelled[1] != "a1",
            "the chosen register collides: {spelled:?}"
        );
    }

    #[test]
    fn riscv_refuses_the_registers_the_frame_is_addressed_from() {
        for name in ["sp", "s0", "fp", "gp", "tp", "zero"] {
            let message = format!("{:?}", named_register(name).expect_err("refused"));
            assert!(message.contains("reserved"), "{name}: {message}");
        }
    }

    #[test]
    fn riscv_refuses_a_register_it_has_never_heard_of() {
        let message = format!("{:?}", named_register("rax").expect_err("refused"));
        assert!(message.contains("unknown register"), "{message}");
    }
}
