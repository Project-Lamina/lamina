//! Checked arithmetic: whether the wrapping result leaves the range of its type.
//!
//! RISC-V reports no flags, so every answer is computed. Below a register the exact
//! result fits, and the question is whether re-reading it at the operand's width gives
//! the same number back. At a register's width addition and subtraction are read from
//! the sign bits: an unsigned sum that came out below one of its operands carried, and
//! a signed one overflowed when both operands shared a sign the result does not.
//! Multiplication takes the upper half of the product, which must be the sign of the
//! lower half for a signed operand and zero for an unsigned one.

use super::util::{load_operand_to_register, store_register_to_register};
use super::{RiscVRegAlloc, RiscVTarget, Xlen};
use crate::{
    error::LaminaError,
    mir::{IntOverflowOp, MirType, Operand, Register, VirtualReg},
};
use lamina_platform::PointerWidth;
use std::collections::HashMap;
use std::io::Write;

const LHS: &str = "t0";
const RHS: &str = "t1";
const RESULT: &str = "t2";
const FLAG: &str = "t3";

fn extend<W: Write>(
    writer: &mut W,
    register: &str,
    bits: usize,
    xlen: usize,
    signed: bool,
) -> Result<(), LaminaError> {
    if bits >= xlen {
        return Ok(());
    }
    let shift = xlen - bits;
    let down = if signed { "srai" } else { "srli" };
    writeln!(writer, "    slli {register}, {register}, {shift}")?;
    writeln!(writer, "    {down} {register}, {register}, {shift}")?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit<W: Write>(
    writer: &mut W,
    reg_alloc: &RiscVRegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    target: RiscVTarget,
    op: &IntOverflowOp,
    ty: &MirType,
    signed: bool,
    dst: &Register,
    lhs: &Operand,
    rhs: &Operand,
) -> Result<(), LaminaError> {
    let xlen = match target.xlen {
        Xlen::Rv32 => 32,
        _ => 64,
    };
    let pointer_width = if xlen == 32 {
        PointerWidth::Bits32
    } else {
        PointerWidth::Bits64
    };
    let bits = ty.size_bytes(pointer_width) * 8;
    let top = xlen - 1;

    load_operand_to_register(lhs, writer, reg_alloc, stack_slots, LHS)?;
    load_operand_to_register(rhs, writer, reg_alloc, stack_slots, RHS)?;

    if bits < xlen {
        extend(writer, LHS, bits, xlen, signed)?;
        extend(writer, RHS, bits, xlen, signed)?;
        let mnemonic = match op {
            IntOverflowOp::Add => "add",
            IntOverflowOp::Sub => "sub",
            IntOverflowOp::Mul => "mul",
        };
        writeln!(writer, "    {mnemonic} {RESULT}, {LHS}, {RHS}")?;
        writeln!(writer, "    mv {FLAG}, {RESULT}")?;
        extend(writer, FLAG, bits, xlen, signed)?;
        writeln!(writer, "    xor {FLAG}, {FLAG}, {RESULT}")?;
        writeln!(writer, "    snez {FLAG}, {FLAG}")?;
    } else {
        match (op, signed) {
            (IntOverflowOp::Add, true) => {
                writeln!(writer, "    add {RESULT}, {LHS}, {RHS}")?;
                writeln!(writer, "    xor {FLAG}, {LHS}, {RHS}")?;
                writeln!(writer, "    xori {FLAG}, {FLAG}, -1")?;
                writeln!(writer, "    xor {LHS}, {RESULT}, {LHS}")?;
                writeln!(writer, "    and {FLAG}, {FLAG}, {LHS}")?;
                writeln!(writer, "    srli {FLAG}, {FLAG}, {top}")?;
            }
            (IntOverflowOp::Add, false) => {
                writeln!(writer, "    add {RESULT}, {LHS}, {RHS}")?;
                writeln!(writer, "    sltu {FLAG}, {RESULT}, {LHS}")?;
            }
            (IntOverflowOp::Sub, true) => {
                writeln!(writer, "    sub {RESULT}, {LHS}, {RHS}")?;
                writeln!(writer, "    xor {FLAG}, {LHS}, {RHS}")?;
                writeln!(writer, "    xor {LHS}, {RESULT}, {LHS}")?;
                writeln!(writer, "    and {FLAG}, {FLAG}, {LHS}")?;
                writeln!(writer, "    srli {FLAG}, {FLAG}, {top}")?;
            }
            (IntOverflowOp::Sub, false) => {
                writeln!(writer, "    sltu {FLAG}, {LHS}, {RHS}")?;
            }
            (IntOverflowOp::Mul, true) => {
                writeln!(writer, "    mul {RESULT}, {LHS}, {RHS}")?;
                writeln!(writer, "    mulh {FLAG}, {LHS}, {RHS}")?;
                writeln!(writer, "    srai {LHS}, {RESULT}, {top}")?;
                writeln!(writer, "    xor {FLAG}, {FLAG}, {LHS}")?;
                writeln!(writer, "    snez {FLAG}, {FLAG}")?;
            }
            (IntOverflowOp::Mul, false) => {
                writeln!(writer, "    mulhu {FLAG}, {LHS}, {RHS}")?;
                writeln!(writer, "    snez {FLAG}, {FLAG}")?;
            }
        }
    }

    match dst {
        Register::Virtual(vreg) => {
            store_register_to_register(FLAG, vreg, writer, reg_alloc, stack_slots)?;
        }
        Register::Physical(physical) => {
            writeln!(writer, "    mv {}, {FLAG}", physical.name)?;
        }
    }
    Ok(())
}
