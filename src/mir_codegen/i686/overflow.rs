//! Checked arithmetic: whether the wrapping result leaves the range of its type.
//!
//! Below a word the exact result fits in a register, so the question is whether
//! re-reading it at the operand's width gives the same number back. At a word the
//! flags answer it directly: addition and subtraction set the overflow flag for a
//! signed operand and the carry flag for an unsigned one, and both multiply
//! instructions set the overflow flag when the product needs more than the low half.

use super::extend_in_place;
use super::operand::{EAX, ECX, EDX, Slots, byte_reg, load_operand, store_register};
use crate::{
    error::LaminaError,
    mir::{IntOverflowOp, MirType, Operand, Register},
};
use lamina_platform::PointerWidth;
use std::io::Write;

const WORD_BYTES: usize = 4;

#[allow(clippy::too_many_arguments)]
pub(super) fn emit<W: Write>(
    writer: &mut W,
    slots: &Slots,
    op: &IntOverflowOp,
    ty: &MirType,
    signed: bool,
    dst: &Register,
    lhs: &Operand,
    rhs: &Operand,
) -> Result<(), LaminaError> {
    load_operand(lhs, writer, slots, EAX)?;
    load_operand(rhs, writer, slots, ECX)?;
    let condition = if ty.size_bytes(PointerWidth::Bits32) < WORD_BYTES {
        extend_in_place(writer, EAX, ty, signed)?;
        extend_in_place(writer, ECX, ty, signed)?;
        let mnemonic = match op {
            IntOverflowOp::Add => "addl",
            IntOverflowOp::Sub => "subl",
            IntOverflowOp::Mul => "imull",
        };
        writeln!(writer, "    {mnemonic} {ECX}, {EAX}")?;
        writeln!(writer, "    movl {EAX}, {EDX}")?;
        extend_in_place(writer, EDX, ty, signed)?;
        writeln!(writer, "    cmpl {EDX}, {EAX}")?;
        "ne"
    } else {
        match (op, signed) {
            (IntOverflowOp::Add, true) => {
                writeln!(writer, "    addl {ECX}, {EAX}")?;
                "o"
            }
            (IntOverflowOp::Add, false) => {
                writeln!(writer, "    addl {ECX}, {EAX}")?;
                "c"
            }
            (IntOverflowOp::Sub, true) => {
                writeln!(writer, "    subl {ECX}, {EAX}")?;
                "o"
            }
            (IntOverflowOp::Sub, false) => {
                writeln!(writer, "    subl {ECX}, {EAX}")?;
                "c"
            }
            (IntOverflowOp::Mul, true) => {
                writeln!(writer, "    imull {ECX}, {EAX}")?;
                "o"
            }
            (IntOverflowOp::Mul, false) => {
                writeln!(writer, "    mull {ECX}")?;
                "o"
            }
        }
    };
    let low = byte_reg(EAX)?;
    writeln!(writer, "    set{condition} {low}")?;
    writeln!(writer, "    movzbl {low}, {EAX}")?;
    store_register(EAX, dst, writer, slots)
}
