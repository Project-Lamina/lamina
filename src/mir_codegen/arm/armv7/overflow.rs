//! Checked arithmetic: whether the wrapping result leaves the range of its type.
//!
//! Below a word the exact result fits in a register, so the question is whether
//! re-reading it at the operand's width gives the same number back. At a word the
//! flags answer it for addition and subtraction, and multiplication takes the long
//! product: unsigned overflows when the high half is nonzero, signed when it is not
//! the sign of the low half.

use super::operand::{Scratch, extend_integer, materialize, store_result};
use crate::{
    error::LaminaError,
    mir::{IntOverflowOp, MirType, Operand, Register},
};
use lamina_codegen::aarch64::FrameMap;
use lamina_platform::PointerWidth;
use std::io::Write;

const WORD_BYTES: usize = 4;

#[allow(clippy::too_many_arguments)]
pub(super) fn emit<W: Write>(
    w: &mut W,
    op: &IntOverflowOp,
    ty: &MirType,
    signed: bool,
    dst: &Register,
    lhs: &Operand,
    rhs: &Operand,
    frame: &FrameMap,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    let a = scratch.alloc()?;
    let b = scratch.alloc()?;
    let flag = scratch.alloc()?;
    let high = scratch.alloc()?;
    materialize(w, lhs, a, frame)?;
    materialize(w, rhs, b, frame)?;
    let condition = if ty.size_bytes(PointerWidth::Bits32) < WORD_BYTES {
        extend_integer(w, a, ty, signed)?;
        extend_integer(w, b, ty, signed)?;
        match op {
            IntOverflowOp::Add => writeln!(w, "    add {a}, {a}, {b}")?,
            IntOverflowOp::Sub => writeln!(w, "    sub {a}, {a}, {b}")?,
            IntOverflowOp::Mul => writeln!(w, "    mul {a}, {a}, {b}")?,
        }
        writeln!(w, "    mov {flag}, {a}")?;
        extend_integer(w, flag, ty, signed)?;
        writeln!(w, "    cmp {a}, {flag}")?;
        "ne"
    } else {
        match (op, signed) {
            (IntOverflowOp::Add, true) => {
                writeln!(w, "    adds {a}, {a}, {b}")?;
                "vs"
            }
            (IntOverflowOp::Add, false) => {
                writeln!(w, "    adds {a}, {a}, {b}")?;
                "cs"
            }
            (IntOverflowOp::Sub, true) => {
                writeln!(w, "    subs {a}, {a}, {b}")?;
                "vs"
            }
            (IntOverflowOp::Sub, false) => {
                writeln!(w, "    subs {a}, {a}, {b}")?;
                "cc"
            }
            (IntOverflowOp::Mul, true) => {
                writeln!(w, "    smull {flag}, {high}, {a}, {b}")?;
                writeln!(w, "    cmp {high}, {flag}, asr #31")?;
                "ne"
            }
            (IntOverflowOp::Mul, false) => {
                writeln!(w, "    umull {flag}, {high}, {a}, {b}")?;
                writeln!(w, "    cmp {high}, #0")?;
                "ne"
            }
        }
    };
    writeln!(w, "    mov {a}, #0")?;
    writeln!(w, "    mov{condition} {a}, #1")?;
    store_result(w, dst, a, frame)?;
    scratch.free(high);
    scratch.free(flag);
    scratch.free(b);
    scratch.free(a);
    Ok(())
}
