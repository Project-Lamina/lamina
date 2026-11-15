//! Lowering of `i128` and `u128` during IR-to-MIR conversion.
//!
//! No target has 128-bit integer registers, so a 128-bit value is two 64-bit words
//! everywhere. A wide value lives in a pair of 64-bit virtual registers, and every
//! operation becomes the sequence that computes both words. The backends and the
//! register allocators only ever see 64-bit values.
//!
//! Values cross a function boundary in memory, as 16-byte integer aggregates. That is
//! the convention the backends already implement, and System V classifies an
//! `__int128` the same way: two INTEGER words, register pair or stack, never split.
//! Division, remainder and float conversion call compiler-rt, through the
//! [`RuntimeSymbol`] seam, with the same aggregate convention.

use super::*;
use crate::ir::instruction::{AsmOperand as IRAsmOperand, UnaryOp};
use crate::ir::types::Literal;
use crate::mir::codegen::mapping::integer_aggregate;
use lamina_platform::{Endianness, TargetDataLayout};

/// How many registers an integer of this type takes. A 128-bit integer is two registers
/// where a register holds 64 bits and four where it holds 32; a 64-bit integer is two
/// only on the narrower target. Everything else fits one register and is not legalized
/// here.
pub(super) fn prim_words(ty: IRPrim, int64_regs: bool) -> Option<usize> {
    match ty {
        IRPrim::I128 | IRPrim::U128 => Some(if int64_regs { 2 } else { 4 }),
        IRPrim::I64 | IRPrim::U64 if !int64_regs => Some(2),
        _ => None,
    }
}

pub(super) fn is_wide_prim(ty: IRPrim, int64_regs: bool) -> bool {
    prim_words(ty, int64_regs).is_some()
}

/// A value wider than a register is moved a word at a time, so the access the
/// instruction asks for is not the access the hardware performs. Rather than let a
/// stated alignment mean nothing, refuse it.
fn reject_wide_alignment(
    align: Option<u32>,
    ty: &IRType<'_>,
    what: &str,
) -> Result<(), FromIRError> {
    match align {
        None => Ok(()),
        Some(align) => Err(FromIRError::UnsupportedAlignment(format!(
            "{what}.{ty} align {align}: a value wider than a register is moved one word at a time, so the stated alignment would not describe the access performed"
        ))),
    }
}

pub(super) fn type_words(ty: &IRType<'_>, int64_regs: bool) -> Option<usize> {
    match ty {
        IRType::Primitive(prim) => prim_words(*prim, int64_regs),
        _ => None,
    }
}

pub(super) fn is_wide_type(ty: &IRType<'_>, int64_regs: bool) -> bool {
    type_words(ty, int64_regs).is_some()
}

pub(super) fn word_ty(int64_regs: bool) -> MirType {
    if int64_regs {
        MirType::Scalar(ScalarType::I64)
    } else {
        MirType::Scalar(ScalarType::I32)
    }
}

const fn word_bits(int64_regs: bool) -> i64 {
    if int64_regs { 64 } else { 32 }
}

const fn word_bytes(int64_regs: bool) -> i32 {
    if int64_regs { 8 } else { 4 }
}

/// Byte offset of word `index` of a value held in memory. The most significant word
/// comes first in big-endian order, so a word's place depends on how many there are.
const fn word_offset(index: usize, count: usize, int64_regs: bool, endianness: Endianness) -> i32 {
    let width = word_bytes(int64_regs);
    let step = match endianness {
        Endianness::Little => index,
        Endianness::Big => count - 1 - index,
    };
    width * step as i32
}

/// The aggregate a legalized integer of `words` registers crosses a function boundary
/// as. A 32-bit target aligns it to a register pair, which is what the C ABIs give a
/// 64- or 128-bit integer there.
pub(super) fn words_aggregate(words: usize, int64_regs: bool) -> Aggregate {
    let size = words * word_bytes(int64_regs) as usize;
    let align = if int64_regs { size } else { size.min(8) };
    integer_aggregate(size, align)
}

fn word_imm(int64_regs: bool, value: i64) -> Operand {
    if int64_regs {
        Operand::Immediate(Immediate::I64(value))
    } else {
        Operand::Immediate(Immediate::I32(value as i32))
    }
}

/// A legalized integer, least significant word first.
#[derive(Clone)]
pub(super) struct Words(Vec<Operand>);

impl Words {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn at(&self, index: usize) -> Operand {
        self.0[index].clone()
    }

    fn lo(&self) -> Operand {
        self.at(0)
    }

    fn top(&self) -> Operand {
        self.at(self.0.len() - 1)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Abi {
    Wide(usize),
    Scalar(MirType),
}

#[derive(Clone, Debug)]
pub(super) struct LibCall {
    name: String,
    params: Vec<Abi>,
    ret: Option<Abi>,
}

impl LibCall {
    pub(super) fn declaration(&self, int64_regs: bool) -> Function {
        let mut sig = Signature::new(self.name.clone());
        for (index, param) in self.params.iter().enumerate() {
            let slot = u32::try_from(index).unwrap_or(u32::MAX);
            let parameter = match param {
                Abi::Wide(words) => Parameter::new(
                    Register::Virtual(VirtualReg::gpr(slot)),
                    MirType::Scalar(ScalarType::Ptr),
                )
                .with_aggregate(words_aggregate(*words, int64_regs)),
                Abi::Scalar(ty) => {
                    let class = if ty.is_float() {
                        RegisterClass::Fpr
                    } else {
                        RegisterClass::Gpr
                    };
                    Parameter::new(Register::Virtual(VirtualReg::new(slot, class)), *ty)
                }
            };
            sig.params.push(parameter);
        }
        match self.ret {
            None => {}
            Some(Abi::Wide(words)) => {
                sig = sig
                    .with_return(MirType::Scalar(ScalarType::Ptr))
                    .with_return_aggregate(words_aggregate(words, int64_regs));
            }
            Some(Abi::Scalar(ty)) => sig = sig.with_return(ty),
        }
        Function::new(sig)
    }
}

pub(super) struct Cx<'c, 'a> {
    pub(super) vregs: &'c mut VirtualRegAllocator,
    pub(super) vars: &'c mut VarBindings<'a>,
    pub(super) reservations: &'c mut HashMap<VirtualReg, usize>,
    pub(super) globals: &'c mut GlobalAddresses<'a>,
    pub(super) module: &'c IRModule<'a>,
    pub(super) layout: TargetDataLayout,
    pub(super) int64_regs: bool,
    pub(super) endianness: Endianness,
    pub(super) pointer_aligns: &'c PointerAligns<'a>,
    pub(super) out: Vec<Instruction>,
}

impl<'c, 'a> Cx<'c, 'a> {
    fn word(&self) -> MirType {
        word_ty(self.int64_regs)
    }

    fn bits(&self) -> i64 {
        word_bits(self.int64_regs)
    }

    fn imm(&self, value: i64) -> Operand {
        word_imm(self.int64_regs, value)
    }

    fn zeros(&self, count: usize) -> Words {
        Words(vec![self.imm(0); count])
    }

    fn temp(&mut self) -> Register {
        Register::Virtual(self.vregs.allocate_gpr())
    }

    fn bin(&mut self, op: IntBinOp, lhs: Operand, rhs: Operand) -> Operand {
        if let (Some(a), Some(b)) = (lhs.integer_constant(), rhs.integer_constant())
            && matches!(lhs, Operand::Immediate(_))
            && matches!(rhs, Operand::Immediate(_))
            && let Some(value) = fold(op, a, b, self.int64_regs)
        {
            return self.imm(value);
        }
        let dst = self.temp();
        let ty = self.word();
        self.out.push(Instruction::IntBinary {
            op,
            ty,
            dst: dst.clone(),
            lhs,
            rhs,
        });
        Operand::Register(dst)
    }

    fn not(&mut self, value: Operand) -> Operand {
        let ones = self.imm(-1);
        self.bin(IntBinOp::Xor, value, ones)
    }

    fn nonzero(&mut self, value: Operand) -> Operand {
        let zero = self.imm(0);
        let negated = self.bin(IntBinOp::Sub, zero, value.clone());
        let any = self.bin(IntBinOp::Or, value, negated);
        let top = self.imm(self.bits() - 1);
        self.bin(IntBinOp::LShr, any, top)
    }

    fn is_zero(&mut self, value: Operand) -> Operand {
        let nonzero = self.nonzero(value);
        let one = self.imm(1);
        self.bin(IntBinOp::Xor, nonzero, one)
    }

    fn carry_out(&mut self, x: Operand, y: Operand, sum: Operand) -> Operand {
        let both = self.bin(IntBinOp::And, x.clone(), y.clone());
        let either = self.bin(IntBinOp::Or, x, y);
        let not_sum = self.not(sum);
        let propagate = self.bin(IntBinOp::And, either, not_sum);
        let joined = self.bin(IntBinOp::Or, both, propagate);
        let top = self.imm(self.bits() - 1);
        self.bin(IntBinOp::LShr, joined, top)
    }

    fn borrow_out(&mut self, x: Operand, y: Operand, diff: Operand) -> Operand {
        let not_x = self.not(x.clone());
        let lent = self.bin(IntBinOp::And, not_x, y.clone());
        let differ = self.bin(IntBinOp::Xor, x, y);
        let same = self.not(differ);
        let propagate = self.bin(IntBinOp::And, same, diff);
        let joined = self.bin(IntBinOp::Or, lent, propagate);
        let top = self.imm(self.bits() - 1);
        self.bin(IntBinOp::LShr, joined, top)
    }

    /// Word-by-word addition, carrying upward. The carry out of the top word is the
    /// overflow of the value as a whole and is dropped.
    fn add(&mut self, a: Words, b: Words) -> Words {
        let count = a.len();
        let mut carry: Option<Operand> = None;
        let mut out = Vec::with_capacity(count);
        for index in 0..count {
            let x = a.at(index);
            let y = b.at(index);
            let sum = self.bin(IntBinOp::Add, x.clone(), y.clone());
            let word = match carry.take() {
                None => sum,
                Some(carried) => self.bin(IntBinOp::Add, sum, carried),
            };
            if index + 1 < count {
                carry = Some(self.carry_out(x, y, word.clone()));
            }
            out.push(word);
        }
        Words(out)
    }

    /// Word-by-word subtraction, borrowing upward. `borrow_out` reads the borrow of a
    /// word that already had the one below taken off it, which is the same expression.
    fn sub(&mut self, a: Words, b: Words) -> Words {
        let count = a.len();
        let mut borrow: Option<Operand> = None;
        let mut out = Vec::with_capacity(count);
        for index in 0..count {
            let x = a.at(index);
            let y = b.at(index);
            let diff = self.bin(IntBinOp::Sub, x.clone(), y.clone());
            let word = match borrow.take() {
                None => diff,
                Some(borrowed) => self.bin(IntBinOp::Sub, diff, borrowed),
            };
            if index + 1 < count {
                borrow = Some(self.borrow_out(x, y, word.clone()));
            }
            out.push(word);
        }
        Words(out)
    }

    fn mul_high_unsigned(&mut self, a: Operand, b: Operand) -> Operand {
        let half = self.bits() / 2;
        let mask = self.imm((1i64 << half) - 1);
        let half = self.imm(half);
        let a0 = self.bin(IntBinOp::And, a.clone(), mask.clone());
        let a1 = self.bin(IntBinOp::LShr, a, half.clone());
        let b0 = self.bin(IntBinOp::And, b.clone(), mask.clone());
        let b1 = self.bin(IntBinOp::LShr, b, half.clone());
        let w0 = self.bin(IntBinOp::Mul, a0.clone(), b0.clone());
        let w0_high = self.bin(IntBinOp::LShr, w0, half.clone());
        let a1b0 = self.bin(IntBinOp::Mul, a1.clone(), b0);
        let t = self.bin(IntBinOp::Add, a1b0, w0_high);
        let t_low = self.bin(IntBinOp::And, t.clone(), mask);
        let t_high = self.bin(IntBinOp::LShr, t, half.clone());
        let a0b1 = self.bin(IntBinOp::Mul, a0, b1.clone());
        let w1 = self.bin(IntBinOp::Add, t_low, a0b1);
        let w1_high = self.bin(IntBinOp::LShr, w1, half);
        let a1b1 = self.bin(IntBinOp::Mul, a1, b1);
        let partial = self.bin(IntBinOp::Add, a1b1, t_high);
        self.bin(IntBinOp::Add, partial, w1_high)
    }

    /// Adds `value` into word `index` of the accumulator, carrying into the words above
    /// it. A word that nothing has reached yet takes the value as it stands.
    fn accumulate(&mut self, acc: &mut [Option<Operand>], index: usize, value: Operand) {
        let mut index = index;
        let mut value = value;
        while index < acc.len() {
            let Some(held) = acc[index].take() else {
                acc[index] = Some(value);
                return;
            };
            let sum = self.bin(IntBinOp::Add, held.clone(), value.clone());
            if index + 1 == acc.len() {
                acc[index] = Some(sum);
                return;
            }
            let carry = self.carry_out(held, value, sum.clone());
            acc[index] = Some(sum);
            value = carry;
            index += 1;
        }
    }

    /// Schoolbook multiplication over the words, keeping the low half of the product.
    /// The high word of a partial product lands one place up, and products that would
    /// land past the top are never computed.
    fn mul(&mut self, a: Words, b: Words) -> Words {
        let count = a.len();
        let mut acc: Vec<Option<Operand>> = vec![None; count];
        for i in 0..count {
            for j in 0..count - i {
                let x = a.at(i);
                let y = b.at(j);
                let low = self.bin(IntBinOp::Mul, x.clone(), y.clone());
                self.accumulate(&mut acc, i + j, low);
                if i + j + 1 < count {
                    let high = self.mul_high_unsigned(x, y);
                    self.accumulate(&mut acc, i + j + 1, high);
                }
            }
        }
        let zero = self.imm(0);
        Words(
            acc.into_iter()
                .map(|word| match word {
                    Some(word) => word,
                    None => zero.clone(),
                })
                .collect(),
        )
    }

    fn bitwise(&mut self, op: IntBinOp, a: Words, b: Words) -> Words {
        let mut out = Vec::with_capacity(a.len());
        for index in 0..a.len() {
            out.push(self.bin(op, a.at(index), b.at(index)));
        }
        Words(out)
    }

    fn complement(&mut self, a: Words) -> Words {
        let mut out = Vec::with_capacity(a.len());
        for index in 0..a.len() {
            out.push(self.not(a.at(index)));
        }
        Words(out)
    }

    /// Reversing the bytes of the value reverses the words as well as the bytes in
    /// each of them.
    fn byteswap(&mut self, a: Words) -> Words {
        let count = a.len();
        let mut out = Vec::with_capacity(count);
        for index in 0..count {
            out.push(self.unary_word(IntUnOp::Bswap, a.at(count - 1 - index)));
        }
        Words(out)
    }

    /// The whole product, twice as many words as the operands. Nothing is dropped, so
    /// the upper half says whether the product fits.
    fn mul_full(&mut self, a: &Words, b: &Words) -> Words {
        let count = a.len();
        let mut acc: Vec<Option<Operand>> = vec![None; 2 * count];
        for i in 0..count {
            for j in 0..count {
                let x = a.at(i);
                let y = b.at(j);
                let low = self.bin(IntBinOp::Mul, x.clone(), y.clone());
                self.accumulate(&mut acc, i + j, low);
                let high = self.mul_high_unsigned(x, y);
                self.accumulate(&mut acc, i + j + 1, high);
            }
        }
        let zero = self.imm(0);
        Words(
            acc.into_iter()
                .map(|word| match word {
                    Some(word) => word,
                    None => zero.clone(),
                })
                .collect(),
        )
    }

    /// The word's sign, as zero or one.
    fn sign_of(&mut self, value: Operand) -> Operand {
        let top = self.imm(self.bits() - 1);
        self.bin(IntBinOp::LShr, value, top)
    }

    /// Whether the two differ, as zero or one, given each is already zero or one.
    fn differs(&mut self, x: Operand, y: Operand) -> Operand {
        self.bin(IntBinOp::Xor, x, y)
    }

    /// Whether the operation on the whole value leaves its width. An unsigned addition
    /// overflows when the top word carries out; a signed one when both operands share
    /// a sign the result does not.
    fn overflow(&mut self, op: IROverflow, signed: bool, a: Words, b: Words) -> Operand {
        let count = a.len();
        match op {
            IROverflow::Add => {
                let sum = self.add(a.clone(), b.clone());
                if !signed {
                    return self.carry_out(a.top(), b.top(), sum.top());
                }
                let sa = self.sign_of(a.top());
                let sb = self.sign_of(b.top());
                let ss = self.sign_of(sum.top());
                let apart = self.differs(sa.clone(), sb);
                let one = self.imm(1);
                let together = self.bin(IntBinOp::Xor, apart, one);
                let moved = self.differs(ss, sa);
                self.bin(IntBinOp::And, together, moved)
            }
            IROverflow::Sub => {
                let diff = self.sub(a.clone(), b.clone());
                if !signed {
                    return self.borrow_out(a.top(), b.top(), diff.top());
                }
                let sa = self.sign_of(a.top());
                let sb = self.sign_of(b.top());
                let sd = self.sign_of(diff.top());
                let apart = self.differs(sa.clone(), sb);
                let moved = self.differs(sd, sa);
                self.bin(IntBinOp::And, apart, moved)
            }
            IROverflow::Mul => {
                let full = self.mul_full(&a, &b);
                let low = Words((0..count).map(|index| full.at(index)).collect());
                let mut high = Words((count..2 * count).map(|index| full.at(index)).collect());
                if signed {
                    // The words above hold the unsigned product. Each operand that is
                    // negative contributes the other one shifted a whole half up, so
                    // taking both off leaves the signed product's upper half.
                    for (sign_of, other) in [(&a, &b), (&b, &a)] {
                        let bit = self.sign_of(sign_of.top());
                        let zero = self.imm(0);
                        let mask = self.bin(IntBinOp::Sub, zero, bit);
                        let masked = Words(
                            (0..count)
                                .map(|index| {
                                    let word = other.at(index);
                                    self.bin(IntBinOp::And, word, mask.clone())
                                })
                                .collect(),
                        );
                        high = self.sub(high, masked);
                    }
                    let top = self.imm(self.bits() - 1);
                    let fill = self.bin(IntBinOp::AShr, low.top(), top);
                    let mut differ: Option<Operand> = None;
                    for index in 0..count {
                        let word = high.at(index);
                        let apart = self.bin(IntBinOp::Xor, word, fill.clone());
                        differ = Some(match differ.take() {
                            None => apart,
                            Some(so_far) => self.bin(IntBinOp::Or, so_far, apart),
                        });
                    }
                    let differ = match differ {
                        Some(word) => word,
                        None => self.imm(0),
                    };
                    return self.nonzero(differ);
                }
                let mut any = high.at(0);
                for index in 1..count {
                    any = self.bin(IntBinOp::Or, any, high.at(index));
                }
                self.nonzero(any)
            }
        }
    }

    /// Splits the shift count into the bits it moves within a word, and the number of
    /// whole words it moves by, as one mask per bit of that number.
    fn shift_parts(&mut self, count: Operand, words: usize) -> (Operand, Vec<Operand>) {
        let width = self.imm(words as i64 * self.bits() - 1);
        let count = self.bin(IntBinOp::And, count, width);
        let within = self.imm(self.bits() - 1);
        let small = self.bin(IntBinOp::And, count.clone(), within);
        let log2 = self.imm(self.bits().trailing_zeros().into());
        let big = self.bin(IntBinOp::LShr, count, log2);
        let mut masks = Vec::new();
        let mut step = 0;
        while 1usize << step < words {
            let shifted = if step == 0 {
                big.clone()
            } else {
                let by = self.imm(step as i64);
                self.bin(IntBinOp::LShr, big.clone(), by)
            };
            let bit = if 1usize << (step + 1) >= words {
                shifted
            } else {
                let one = self.imm(1);
                self.bin(IntBinOp::And, shifted, one)
            };
            let zero = self.imm(0);
            masks.push(self.bin(IntBinOp::Sub, zero, bit));
            step += 1;
        }
        (small, masks)
    }

    /// Bits of the word below that move up for a left shift within a word. Written as
    /// two shifts, so a count of zero contributes nothing and no shift reaches the
    /// register's width.
    fn cross_shift_left(&mut self, low: Operand, count: Operand) -> Operand {
        let one = self.imm(1);
        let halved = self.bin(IntBinOp::LShr, low, one);
        let top = self.imm(self.bits() - 1);
        let rest = self.bin(IntBinOp::Xor, count, top);
        self.bin(IntBinOp::LShr, halved, rest)
    }

    /// Bits of the word above that move down for a right shift within a word, with the
    /// same two-shift form.
    fn cross_shift_right(&mut self, high: Operand, count: Operand) -> Operand {
        let one = self.imm(1);
        let doubled = self.bin(IntBinOp::Shl, high, one);
        let top = self.imm(self.bits() - 1);
        let rest = self.bin(IntBinOp::Xor, count, top);
        self.bin(IntBinOp::Shl, doubled, rest)
    }

    fn select_mask(&mut self, mask: Operand, when_set: Operand, when_clear: Operand) -> Operand {
        let not_mask = self.not(mask.clone());
        let set = self.bin(IntBinOp::And, when_set, mask);
        let clear = self.bin(IntBinOp::And, when_clear, not_mask);
        self.bin(IntBinOp::Or, set, clear)
    }

    /// One round of the word-granular part of a shift: every word takes the word that
    /// is `step` places away when the mask is set, and keeps its own when it is clear.
    /// Words shifted in from outside the value take `fill`, which is nothing for a
    /// logical shift and the sign for an arithmetic one.
    fn shift_words(
        &mut self,
        value: Vec<Operand>,
        mask: Operand,
        step: usize,
        left: bool,
        fill: Option<Operand>,
    ) -> Vec<Operand> {
        let count = value.len();
        let not_mask = self.not(mask.clone());
        let mut out = Vec::with_capacity(count);
        for index in 0..count {
            let source = if left {
                index.checked_sub(step)
            } else {
                let from = index + step;
                (from < count).then_some(from)
            };
            let incoming = match source {
                Some(from) => Some(value[from].clone()),
                None => fill.clone(),
            };
            let kept = value[index].clone();
            out.push(match incoming {
                None => self.bin(IntBinOp::And, kept, not_mask.clone()),
                Some(incoming) => self.select_mask(mask.clone(), incoming, kept),
            });
        }
        out
    }

    fn shl(&mut self, a: Words, count: Operand) -> Words {
        let words = a.len();
        let (small, masks) = self.shift_parts(count, words);
        let mut value: Vec<Operand> = Vec::with_capacity(words);
        for index in 0..words {
            let shifted = self.bin(IntBinOp::Shl, a.at(index), small.clone());
            value.push(match index {
                0 => shifted,
                _ => {
                    let crossed = self.cross_shift_left(a.at(index - 1), small.clone());
                    self.bin(IntBinOp::Or, shifted, crossed)
                }
            });
        }
        for (step, mask) in masks.into_iter().enumerate() {
            value = self.shift_words(value, mask, 1 << step, true, None);
        }
        Words(value)
    }

    fn right_shift(&mut self, a: Words, count: Operand, arithmetic: bool) -> Words {
        let words = a.len();
        let (small, masks) = self.shift_parts(count, words);
        let high_op = if arithmetic {
            IntBinOp::AShr
        } else {
            IntBinOp::LShr
        };
        let fill = if arithmetic {
            let top = self.imm(self.bits() - 1);
            Some(self.bin(IntBinOp::AShr, a.top(), top))
        } else {
            None
        };
        let mut value: Vec<Operand> = Vec::with_capacity(words);
        for index in 0..words {
            if index + 1 == words {
                value.push(self.bin(high_op, a.at(index), small.clone()));
                continue;
            }
            let shifted = self.bin(IntBinOp::LShr, a.at(index), small.clone());
            let crossed = self.cross_shift_right(a.at(index + 1), small.clone());
            value.push(self.bin(IntBinOp::Or, shifted, crossed));
        }
        for (step, mask) in masks.into_iter().enumerate() {
            value = self.shift_words(value, mask, 1 << step, false, fill.clone());
        }
        Words(value)
    }

    fn unary_word(&mut self, op: IntUnOp, src: Operand) -> Operand {
        let dst = self.temp();
        let ty = self.word();
        self.out.push(Instruction::IntUnary {
            op,
            ty,
            dst: dst.clone(),
            src,
        });
        Operand::Register(dst)
    }

    fn popcount(&mut self, a: Words) -> Words {
        let mut total = self.unary_word(IntUnOp::Popcount, a.at(0));
        for index in 1..a.len() {
            let next = self.unary_word(IntUnOp::Popcount, a.at(index));
            total = self.bin(IntBinOp::Add, total, next);
        }
        let mut out = self.zeros(a.len());
        out.0[0] = total;
        out
    }

    /// The count of the topmost word that is not zero, plus a whole word for each zero
    /// word above it. A word's count only counts while every word above it was zero.
    fn leading_zeros(&mut self, a: Words) -> Words {
        let count = a.len();
        let mut total = self.unary_word(IntUnOp::Ctlz, a.at(count - 1));
        let mut empty: Option<Operand> = None;
        for index in (0..count - 1).rev() {
            let above = a.at(index + 1);
            let is_zero = self.is_zero(above);
            let so_far = match empty.take() {
                None => is_zero,
                Some(so_far) => self.bin(IntBinOp::And, so_far, is_zero),
            };
            empty = Some(so_far.clone());
            let zero = self.imm(0);
            let mask = self.bin(IntBinOp::Sub, zero, so_far);
            let here = self.unary_word(IntUnOp::Ctlz, a.at(index));
            let counted = self.bin(IntBinOp::And, here, mask);
            total = self.bin(IntBinOp::Add, total, counted);
        }
        let mut out = self.zeros(count);
        out.0[0] = total;
        out
    }

    /// The same from the other end: the lowest word's count, plus a whole word for each
    /// zero word below the first that is not zero.
    fn trailing_zeros(&mut self, a: Words) -> Words {
        let count = a.len();
        let mut total = self.unary_word(IntUnOp::Cttz, a.at(0));
        let mut empty: Option<Operand> = None;
        for index in 1..count {
            let below = a.at(index - 1);
            let is_zero = self.is_zero(below);
            let so_far = match empty.take() {
                None => is_zero,
                Some(so_far) => self.bin(IntBinOp::And, so_far, is_zero),
            };
            empty = Some(so_far.clone());
            let zero = self.imm(0);
            let mask = self.bin(IntBinOp::Sub, zero, so_far);
            let here = self.unary_word(IntUnOp::Cttz, a.at(index));
            let counted = self.bin(IntBinOp::And, here, mask);
            total = self.bin(IntBinOp::Add, total, counted);
        }
        let mut out = self.zeros(count);
        out.0[0] = total;
        out
    }

    /// Rotation by `count` modulo the value's width. The shifts mask their count, so
    /// the opposite shift takes `-count`, and a zero count leaves the value unchanged.
    fn rotate(&mut self, a: Words, count: Operand, left: bool) -> Words {
        let zero = self.imm(0);
        let opposite = self.bin(IntBinOp::Sub, zero, count.clone());
        let (toward, away) = if left {
            (count, opposite)
        } else {
            (opposite, count)
        };
        let up = self.shl(a.clone(), toward);
        let down = self.right_shift(a, away, false);
        let mut out = Vec::with_capacity(up.len());
        for index in 0..up.len() {
            out.push(self.bin(IntBinOp::Or, up.at(index), down.at(index)));
        }
        Words(out)
    }

    fn unsigned_less(&mut self, a: Words, b: Words) -> Operand {
        let count = a.len();
        let mut borrow: Option<Operand> = None;
        for index in 0..count {
            let x = a.at(index);
            let y = b.at(index);
            let diff = self.bin(IntBinOp::Sub, x.clone(), y.clone());
            let word = match borrow.take() {
                None => diff,
                Some(borrowed) => self.bin(IntBinOp::Sub, diff, borrowed),
            };
            borrow = Some(self.borrow_out(x, y, word));
        }
        match borrow {
            Some(flag) => flag,
            None => self.imm(0),
        }
    }

    fn flip_sign(&mut self, a: Words) -> Words {
        let sign = self.imm(sign_bit(self.int64_regs));
        let count = a.len();
        let mut out = a;
        let top = out.at(count - 1);
        let flipped = self.bin(IntBinOp::Xor, top, sign);
        out.0[count - 1] = flipped;
        out
    }

    /// The address `offset` bytes past `base`, as a register, so the access itself
    /// uses offset zero, the only offset every backend accepts.
    fn word_address(&mut self, base: &Register, offset: i32) -> Register {
        let dst = self.temp();
        self.out.push(Instruction::Lea {
            dst: dst.clone(),
            base: base.clone(),
            offset,
        });
        dst
    }

    fn store_words(&mut self, base: Register, words: Words) {
        self.store_words_at(base, words, self.word_align());
    }

    /// The alignment each word of a wide value is moved at: a word, unless the address
    /// itself is known to fall short of one.
    fn word_align(&self) -> u8 {
        word_bytes(self.int64_regs) as u8
    }

    fn address_align(&self, ptr: &IRVal<'a>) -> u8 {
        pointer_align(ptr, self.pointer_aligns, self.module, self.layout)
    }

    fn store_words_at(&mut self, base: Register, words: Words, align: u8) {
        let attrs = MemoryAttrs {
            align: align.min(self.word_align()),
            volatile: false,
            stated_align: false,
        };
        let ty = self.word();
        let count = words.len();
        for index in 0..count {
            let offset = word_offset(index, count, self.int64_regs, self.endianness);
            let src = words.at(index);
            let base = if offset == 0 {
                base.clone()
            } else {
                self.word_address(&base, offset)
            };
            self.out.push(Instruction::Store {
                ty,
                src,
                addr: AddressMode::BaseOffset {
                    base,
                    offset: 0,
                    address_space: Default::default(),
                },
                attrs,
            });
        }
    }

    fn load_words(&mut self, base: Register, count: usize) -> Words {
        self.load_words_at(base, count, self.word_align())
    }

    fn load_words_at(&mut self, base: Register, count: usize, align: u8) -> Words {
        let attrs = MemoryAttrs {
            align: align.min(self.word_align()),
            volatile: false,
            stated_align: false,
        };
        let ty = self.word();
        let mut out = Vec::with_capacity(count);
        for index in 0..count {
            let offset = word_offset(index, count, self.int64_regs, self.endianness);
            let base = if offset == 0 {
                base.clone()
            } else {
                self.word_address(&base, offset)
            };
            let dst = self.temp();
            self.out.push(Instruction::Load {
                ty,
                dst: dst.clone(),
                addr: AddressMode::BaseOffset {
                    base,
                    offset: 0,
                    address_space: Default::default(),
                },
                attrs,
            });
            out.push(Operand::Register(dst));
        }
        Words(out)
    }

    fn spill(&mut self, words: Words) -> Operand {
        let address = self.words_slot(words.len());
        self.store_words(address.clone(), words);
        Operand::Register(address)
    }

    /// Records that `address` holds a legalized integer a call passes where the
    /// callee's declaration gives no type for it, so a backend can place it as the
    /// callee reads it. A 128-bit integer in a variadic tail has no lowering on a
    /// target whose registers already hold half of it, so that stops the compile by
    /// name.
    fn mark_variadic_wide(
        &mut self,
        address: &Operand,
        words: usize,
        callee: &str,
    ) -> Result<(), FromIRError> {
        if self.int64_regs {
            return Err(FromIRError::NotLegalizable(format!(
                "a 128-bit integer passed to '@{callee}' past its named parameters: no backend places one in a variadic tail"
            )));
        }
        let Operand::Register(Register::Virtual(vreg)) = address else {
            return Err(FromIRError::InvalidIR);
        };
        self.vars
            .variadic_wide
            .insert(*vreg, words_aggregate(words, self.int64_regs));
        Ok(())
    }

    fn libcall_name(
        &mut self,
        symbol: RuntimeSymbol,
        params: Vec<Abi>,
        ret: Option<Abi>,
    ) -> String {
        let name = symbol.c_name().to_string();
        self.vars
            .libcalls
            .entry(name.clone())
            .or_insert_with(|| LibCall {
                name: name.clone(),
                params,
                ret,
            });
        name
    }

    fn call_wide_result(&mut self, name: String, args: Vec<Operand>, words: usize) -> Words {
        let slot = self.vregs.allocate_gpr();
        self.reservations
            .insert(slot, slot_reservation(words, self.int64_regs));
        self.out.push(Instruction::Call {
            name,
            args,
            ret: Some(Register::Virtual(slot)),
        });
        let address = self.temp();
        self.out.push(Instruction::StackAddr {
            dst: address.clone(),
            slot,
        });
        self.load_words(address, words)
    }

    fn word_of(
        &mut self,
        ty: IRPrim,
        operand: Operand,
        signed: bool,
    ) -> Result<Operand, FromIRError> {
        let from = map_ir_prim(ty, self.layout)?;
        let to = self.word();
        if from == to {
            return Ok(operand);
        }
        let dst = self.temp();
        self.out.push(Instruction::IntConvert {
            from,
            to,
            signed,
            dst: dst.clone(),
            src: operand,
        });
        Ok(Operand::Register(dst))
    }

    /// The operand a narrow instruction reads from this value: its low word when the
    /// value is legalized, otherwise the value itself.
    fn narrow(&mut self, value: &IRVal<'a>) -> Result<Operand, FromIRError> {
        if let Some(words) = value_words(value, self.vars, self.int64_regs) {
            return Ok(self.operand_words(value, words)?.lo());
        }
        resolve_operand(value, self.vregs, self.vars, self.globals)
    }

    /// A stack slot deep enough for a legalized integer, as a register holding its
    /// address.
    fn words_slot(&mut self, words: usize) -> Register {
        let slot = self.vregs.allocate_gpr();
        self.reservations
            .insert(slot, slot_reservation(words, self.int64_regs));
        let address = self.temp();
        self.out.push(Instruction::StackAddr {
            dst: address.clone(),
            slot,
        });
        address
    }

    fn operand_words(&mut self, value: &IRVal<'a>, words: usize) -> Result<Words, FromIRError> {
        match value {
            IRVal::Constant(literal) => {
                let parts = literal_words(literal, words, self.int64_regs)?;
                Ok(Words(
                    parts.into_iter().map(|word| self.imm(word)).collect(),
                ))
            }
            IRVal::Variable(name) => {
                let name: &'a str = name;
                if let Some(held) = self.vars.wide.get(name).cloned() {
                    if held.len() != words {
                        return Err(FromIRError::UnsupportedType);
                    }
                    return Ok(Words(held.into_iter().map(Operand::Register).collect()));
                }
                if self.vars.contains_key(name) {
                    return Err(FromIRError::UnsupportedType);
                }
                let held: Vec<Register> = (0..words)
                    .map(|_| Register::Virtual(self.vregs.allocate_gpr()))
                    .collect();
                self.vars.wide.insert(name, held.clone());
                Ok(Words(held.into_iter().map(Operand::Register).collect()))
            }
            IRVal::Global(_) => Err(FromIRError::UnsupportedType),
        }
    }

    fn result_words(&mut self, name: &'a str, words: usize) -> Vec<Register> {
        if let Some(held) = self.vars.wide.get(name).cloned() {
            return held;
        }
        let held: Vec<Register> = (0..words).map(|_| self.temp()).collect();
        self.vars.wide.insert(name, held.clone());
        held
    }

    fn write_words(&mut self, into: Vec<Register>, words: Words) {
        let ty = self.word();
        for (index, dst) in into.into_iter().enumerate() {
            let src = words.at(index);
            self.out.push(Instruction::Copy { ty, dst, src });
        }
    }

    fn division(&mut self, signed: bool, remainder: bool, a: Words, b: Words) -> Words {
        let count = a.len();
        let mut joined = b.at(0);
        for index in 1..count {
            joined = self.bin(IntBinOp::Or, joined, b.at(index));
        }
        let nonzero_divisor = self.nonzero(joined);
        let divisor = if signed {
            let sign = self.imm(sign_bit(self.int64_regs));
            let sign_flipped = self.bin(IntBinOp::Xor, a.top(), sign);
            let mut a_is_min = self.is_zero(sign_flipped);
            for index in 0..count - 1 {
                let low = self.is_zero(a.at(index));
                a_is_min = self.bin(IntBinOp::And, a_is_min, low);
            }
            let mut b_is_ones: Option<Operand> = None;
            for index in 0..count {
                let ones = self.imm(-1);
                let flipped = self.bin(IntBinOp::Xor, b.at(index), ones);
                let is_ones = self.is_zero(flipped);
                b_is_ones = Some(match b_is_ones.take() {
                    None => is_ones,
                    Some(so_far) => self.bin(IntBinOp::And, so_far, is_ones),
                });
            }
            let b_is_ones = match b_is_ones {
                Some(flag) => flag,
                None => self.imm(0),
            };
            let overflow = self.bin(IntBinOp::And, a_is_min, b_is_ones);
            let one = self.imm(1);
            let fine = self.bin(IntBinOp::Xor, overflow, one);
            self.bin(IntBinOp::And, nonzero_divisor, fine)
        } else {
            nonzero_divisor
        };
        let trap_op = if signed {
            IntBinOp::SDiv
        } else {
            IntBinOp::UDiv
        };
        let trap_dst = self.temp();
        let ty = self.word();
        let one = self.imm(1);
        self.out.push(Instruction::IntBinary {
            op: trap_op,
            ty,
            dst: trap_dst,
            lhs: one,
            rhs: divisor,
        });

        let quad = count as i64 * self.bits() == 128;
        let symbol = match (quad, signed, remainder) {
            (true, true, false) => RuntimeSymbol::Divti3,
            (true, false, false) => RuntimeSymbol::Udivti3,
            (true, true, true) => RuntimeSymbol::Modti3,
            (true, false, true) => RuntimeSymbol::Umodti3,
            (false, true, false) => RuntimeSymbol::Divdi3,
            (false, false, false) => RuntimeSymbol::Udivdi3,
            (false, true, true) => RuntimeSymbol::Moddi3,
            (false, false, true) => RuntimeSymbol::Umoddi3,
        };
        let name = self.libcall_name(
            symbol,
            vec![Abi::Wide(count), Abi::Wide(count)],
            Some(Abi::Wide(count)),
        );
        let pa = self.spill(a);
        let pb = self.spill(b);
        self.call_wide_result(name, vec![pa, pb], count)
    }
}

/// Stack slots are counted in eight-byte units past the first, so a legalized integer
/// of any width this lowering builds asks for one more than it starts with.
fn slot_reservation(words: usize, int64_regs: bool) -> usize {
    let bytes = words * word_bytes(int64_regs) as usize;
    bytes.div_ceil(8).max(2) - 1
}

/// Whether a value stands for a legalized integer, and of how many words. A literal is
/// only self-evidently wide at 128 bits; at 64 bits on a 32-bit target the literal's
/// own type says nothing, so the callee signature decides and a bare literal stays
/// scalar here.
fn value_words(value: &IRVal<'_>, vars: &VarBindings<'_>, int64_regs: bool) -> Option<usize> {
    match value {
        IRVal::Variable(name) => vars.wide.get(name).map(Vec::len),
        IRVal::Constant(Literal::I128(_) | Literal::U128(_)) => {
            Some(if int64_regs { 2 } else { 4 })
        }
        IRVal::Constant(_) | IRVal::Global(_) => None,
    }
}

/// The value of the word's sign bit, as the word-width immediate that carries it.
const fn sign_bit(int64_regs: bool) -> i64 {
    if int64_regs {
        i64::MIN
    } else {
        i32::MIN as i64
    }
}

/// Folds an operation on two literal words at the word's width, with the same wrapping
/// and masking the backends use. Operations that can trap are left to run.
fn fold(op: IntBinOp, a: i64, b: i64, int64_regs: bool) -> Option<i64> {
    if !int64_regs {
        let (a, b) = (a as i32, b as i32);
        let count = b as u32 & 31;
        return Some(i64::from(match op {
            IntBinOp::Add => a.wrapping_add(b),
            IntBinOp::Sub => a.wrapping_sub(b),
            IntBinOp::Mul => a.wrapping_mul(b),
            IntBinOp::And => a & b,
            IntBinOp::Or => a | b,
            IntBinOp::Xor => a ^ b,
            IntBinOp::Shl => ((a as u32) << count) as i32,
            IntBinOp::LShr => ((a as u32) >> count) as i32,
            IntBinOp::AShr => a >> count,
            IntBinOp::UDiv | IntBinOp::SDiv | IntBinOp::URem | IntBinOp::SRem => return None,
        }));
    }
    let count = (b as u64 & 63) as u32;
    Some(match op {
        IntBinOp::Add => a.wrapping_add(b),
        IntBinOp::Sub => a.wrapping_sub(b),
        IntBinOp::Mul => a.wrapping_mul(b),
        IntBinOp::And => a & b,
        IntBinOp::Or => a | b,
        IntBinOp::Xor => a ^ b,
        IntBinOp::Shl => ((a as u64) << count) as i64,
        IntBinOp::LShr => ((a as u64) >> count) as i64,
        IntBinOp::AShr => a >> count,
        IntBinOp::UDiv | IntBinOp::SDiv | IntBinOp::URem | IntBinOp::SRem => return None,
    })
}

/// The words a literal supplies to a legalized integer of `words` registers, least
/// significant first.
fn literal_words(
    literal: &Literal<'_>,
    words: usize,
    int64_regs: bool,
) -> Result<Vec<i64>, FromIRError> {
    let value: i128 = match literal {
        Literal::I128(value) => *value,
        Literal::U128(value) => *value as i128,
        Literal::I8(value) => i128::from(*value),
        Literal::I16(value) => i128::from(*value),
        Literal::I32(value) => i128::from(*value),
        Literal::I64(value) => i128::from(*value),
        Literal::U8(value) => i128::from(*value),
        Literal::U16(value) => i128::from(*value),
        Literal::U32(value) => i128::from(*value),
        Literal::U64(value) => i128::from(*value),
        Literal::Bool(value) => i128::from(*value),
        Literal::F32(_) | Literal::F64(_) | Literal::String(_) => {
            return Err(FromIRError::UnsupportedType);
        }
    };
    Ok(split(value as u128, words, int64_regs))
}

fn split(value: u128, words: usize, int64_regs: bool) -> Vec<i64> {
    let bits = word_bits(int64_regs) as u32;
    (0..words)
        .map(|index| {
            let moved = index as u32 * bits;
            let shifted = if moved >= 128 { 0 } else { value >> moved };
            if bits == 64 {
                shifted as u64 as i64
            } else {
                i64::from(shifted as u32 as i32)
            }
        })
        .collect()
}

/// The register count of a type the caller has already established is legalized.
fn wide_width(ty: IRPrim, int64_regs: bool) -> Result<usize, FromIRError> {
    prim_words(ty, int64_regs).ok_or(FromIRError::UnsupportedType)
}

fn wide_type_width(ty: &IRType<'_>, int64_regs: bool) -> Result<usize, FromIRError> {
    type_words(ty, int64_regs).ok_or(FromIRError::UnsupportedType)
}

fn is_wide_value(value: &IRVal<'_>, vars: &VarBindings<'_>, int64_regs: bool) -> bool {
    value_words(value, vars, int64_regs).is_some()
}

/// Whether the instruction names a 128-bit type, or reads a 128-bit value. Every variant
/// is listed, so a new IR instruction forces a decision here.
fn touches_wide(
    instr: &IRInst<'_>,
    vars: &VarBindings<'_>,
    module: &IRModule<'_>,
    int64_regs: bool,
) -> bool {
    let typed = match instr {
        IRInst::Binary { ty, .. }
        | IRInst::OverflowFlag { ty, .. }
        | IRInst::Unary { ty, .. }
        | IRInst::Cmp { ty, .. } => is_wide_prim(*ty, int64_regs),
        IRInst::ZeroExtend {
            source_type,
            target_type,
            ..
        }
        | IRInst::SignExtend {
            source_type,
            target_type,
            ..
        }
        | IRInst::Trunc {
            source_type,
            target_type,
            ..
        }
        | IRInst::Bitcast {
            source_type,
            target_type,
            ..
        }
        | IRInst::FloatConvert {
            source_type,
            target_type,
            ..
        } => is_wide_prim(*source_type, int64_regs) || is_wide_prim(*target_type, int64_regs),
        IRInst::IntToFloat { source_type, .. } => is_wide_prim(*source_type, int64_regs),
        IRInst::FloatToInt { target_type, .. }
        | IRInst::PtrToInt { target_type, .. }
        | IRInst::IntToPtr { target_type, .. } => is_wide_prim(*target_type, int64_regs),
        IRInst::Switch { ty, .. } => is_wide_prim(*ty, int64_regs),
        IRInst::Select { ty, .. }
        | IRInst::Ret { ty, .. }
        | IRInst::Copy { ty, .. }
        | IRInst::Load { ty, .. }
        | IRInst::Store { ty, .. }
        | IRInst::AtomicLoad { ty, .. }
        | IRInst::AtomicStore { ty, .. }
        | IRInst::AtomicBinary { ty, .. }
        | IRInst::AtomicCompareExchange { ty, .. } => is_wide_type(ty, int64_regs),
        IRInst::VaArg { ty, .. } => is_wide_prim(*ty, int64_regs),
        IRInst::InlineAsm {
            result, operands, ..
        } => {
            result.as_ref().is_some_and(|(_, ty)| is_wide_prim(*ty, int64_regs))
                || operands.iter().any(|operand| {
                    matches!(operand, IRAsmOperand::In { value, .. } if is_wide_value(value, vars, int64_regs))
                })
        }
        IRInst::CallIndirect {
            args, return_type, ..
        } => is_wide_type(return_type, int64_regs) || args.iter().any(|(ty, _)| is_wide_type(ty, int64_regs)),
        IRInst::Call {
            func_name, args, ..
        } => {
            module.functions.get(func_name).is_some_and(|callee| {
                is_wide_type(&callee.signature.return_type, int64_regs)
                    || callee.signature.params.iter().any(|p| is_wide_type(&p.ty, int64_regs))
            }) || args.iter().any(|(ty, arg)| {
                ty.as_ref().is_some_and(|ty| is_wide_type(ty, int64_regs))
                    || is_wide_value(arg, vars, int64_regs)
            })
        }
        IRInst::Write { .. }
        | IRInst::Read { .. }
        | IRInst::WriteByte { .. }
        | IRInst::ReadByte { .. } => !int64_regs,
        IRInst::Alloc { .. }
        | IRInst::FuncAddr { .. }
        | IRInst::Br { .. }
        | IRInst::Jmp { .. }
        | IRInst::Unreachable
        | IRInst::Fence { .. }
        | IRInst::GetFieldPtr { .. }
        | IRInst::GetElemPtr { .. }
        | IRInst::VaStart { .. }
        | IRInst::VaEnd { .. }
        | IRInst::VaCopy { .. } => false,
        IRInst::VectorBinary { .. }
        | IRInst::VectorSplat { .. }
        | IRInst::VectorExtractLane { .. }
        | IRInst::VectorInsertLane { .. }
        | IRInst::VectorShuffle { .. } => false,
    };
    typed
        || instr
            .used_vars()
            .iter()
            .any(|name| vars.wide.contains_key(name))
}

pub(super) fn lower_instruction<'a>(
    instr: &IRInst<'a>,
    cx: &mut Cx<'_, 'a>,
) -> Result<bool, FromIRError> {
    if !touches_wide(instr, cx.vars, cx.module, cx.int64_regs) {
        return Ok(false);
    }
    match instr {
        IRInst::Binary {
            op,
            result,
            ty,
            lhs,
            rhs,
        } if is_wide_prim(*ty, cx.int64_regs) => {
            let words = wide_width(*ty, cx.int64_regs)?;
            let signed = matches!(*ty, IRPrim::I128 | IRPrim::I64);
            let a = cx.operand_words(lhs, words)?;
            let b = cx.operand_words(rhs, words)?;
            let value = match op {
                IRBin::Add => cx.add(a, b),
                IRBin::Sub => cx.sub(a, b),
                IRBin::Mul => cx.mul(a, b),
                IRBin::And => cx.bitwise(IntBinOp::And, a, b),
                IRBin::Or => cx.bitwise(IntBinOp::Or, a, b),
                IRBin::Xor => cx.bitwise(IntBinOp::Xor, a, b),
                IRBin::Shl => {
                    let count = b.lo();
                    cx.shl(a, count)
                }
                IRBin::Shr => {
                    let count = b.lo();
                    cx.right_shift(a, count, signed)
                }
                IRBin::Div => cx.division(signed, false, a, b),
                IRBin::Rem => cx.division(signed, true, a, b),
                IRBin::Rotl => {
                    let count = b.lo();
                    cx.rotate(a, count, true)
                }
                IRBin::Rotr => {
                    let count = b.lo();
                    cx.rotate(a, count, false)
                }
            };
            let into = cx.result_words(result, words);
            cx.write_words(into, value);
        }
        IRInst::Unary {
            op,
            result,
            ty,
            value,
        } if is_wide_prim(*ty, cx.int64_regs) => {
            let words = wide_width(*ty, cx.int64_regs)?;
            let a = cx.operand_words(value, words)?;
            let value = match op {
                UnaryOp::Neg => {
                    let zero = cx.zeros(words);
                    cx.sub(zero, a)
                }
                UnaryOp::Not => cx.complement(a),
                UnaryOp::Popcount => cx.popcount(a),
                UnaryOp::Ctlz => cx.leading_zeros(a),
                UnaryOp::Cttz => cx.trailing_zeros(a),
                UnaryOp::Bswap => cx.byteswap(a),
                UnaryOp::Sqrt => return Err(FromIRError::UnsupportedInstruction),
            };
            let into = cx.result_words(result, words);
            cx.write_words(into, value);
        }
        IRInst::Cmp {
            op,
            result,
            ty,
            lhs,
            rhs,
        } if is_wide_prim(*ty, cx.int64_regs) => {
            let words = wide_width(*ty, cx.int64_regs)?;
            let signed = matches!(*ty, IRPrim::I128 | IRPrim::I64);
            let a = cx.operand_words(lhs, words)?;
            let b = cx.operand_words(rhs, words)?;
            let dst = resolve_or_alloc(result, cx.vregs, cx.vars);
            let flag = match op {
                IRCmp::Eq | IRCmp::Ne => {
                    let parts = cx.bitwise(IntBinOp::Xor, a, b);
                    let mut differ = parts.lo();
                    for index in 1..words {
                        differ = cx.bin(IntBinOp::Or, differ, parts.at(index));
                    }
                    let op = if matches!(op, IRCmp::Eq) {
                        IntCmpOp::Eq
                    } else {
                        IntCmpOp::Ne
                    };
                    let ty = cx.word();
                    let zero = cx.imm(0);
                    cx.out.push(Instruction::IntCmp {
                        op,
                        ty,
                        dst,
                        lhs: differ,
                        rhs: zero,
                    });
                    return Ok(true);
                }
                IRCmp::Lt | IRCmp::Le | IRCmp::Gt | IRCmp::Ge => {
                    let (x, y) = if signed {
                        (cx.flip_sign(a), cx.flip_sign(b))
                    } else {
                        (a, b)
                    };
                    match op {
                        IRCmp::Lt => cx.unsigned_less(x, y),
                        IRCmp::Gt => cx.unsigned_less(y, x),
                        IRCmp::Le => {
                            let greater = cx.unsigned_less(y, x);
                            let one = cx.imm(1);
                            cx.bin(IntBinOp::Xor, greater, one)
                        }
                        IRCmp::Ge => {
                            let less = cx.unsigned_less(x, y);
                            let one = cx.imm(1);
                            cx.bin(IntBinOp::Xor, less, one)
                        }
                        IRCmp::Eq | IRCmp::Ne => return Err(FromIRError::UnsupportedInstruction),
                    }
                }
            };
            let ty = cx.word();
            let zero = cx.imm(0);
            cx.out.push(Instruction::IntCmp {
                op: IntCmpOp::Ne,
                ty,
                dst,
                lhs: flag,
                rhs: zero,
            });
        }
        IRInst::ZeroExtend {
            result,
            source_type,
            target_type,
            value,
        } if is_wide_prim(*target_type, cx.int64_regs) => {
            let words = wide_width(*target_type, cx.int64_regs)?;
            widen(cx, result, *source_type, value, false, words)?;
        }
        IRInst::SignExtend {
            result,
            source_type,
            target_type,
            value,
        } if is_wide_prim(*target_type, cx.int64_regs) => {
            let words = wide_width(*target_type, cx.int64_regs)?;
            widen(cx, result, *source_type, value, true, words)?;
        }
        IRInst::Trunc {
            result,
            source_type,
            target_type,
            value,
        } if is_wide_prim(*source_type, cx.int64_regs) => {
            let words = wide_width(*source_type, cx.int64_regs)?;
            let a = cx.operand_words(value, words)?;
            if let Some(kept) = prim_words(*target_type, cx.int64_regs) {
                if kept >= words {
                    return Err(FromIRError::UnsupportedInstruction);
                }
                let value = Words((0..kept).map(|index| a.at(index)).collect());
                let into = cx.result_words(result, kept);
                cx.write_words(into, value);
                return Ok(true);
            }
            let dst = resolve_or_alloc(result, cx.vregs, cx.vars);
            let word = cx.word();
            let to = map_ir_prim(*target_type, cx.layout)?;
            if to == word {
                cx.out.push(Instruction::Copy {
                    ty: word,
                    dst,
                    src: a.lo(),
                });
            } else {
                cx.out.push(Instruction::IntConvert {
                    from: word,
                    to,
                    signed: false,
                    dst,
                    src: a.lo(),
                });
            }
        }
        IRInst::Select {
            result,
            ty,
            cond,
            true_val,
            false_val,
        } => {
            let Some(words) = type_words(ty, cx.int64_regs) else {
                return Ok(false);
            };
            let cond_reg = match cond {
                IRVal::Variable(id) => cx
                    .vars
                    .get(id)
                    .cloned()
                    .ok_or(FromIRError::UnknownVariable)?,
                IRVal::Constant(_) | IRVal::Global(_) => {
                    return Err(FromIRError::UnsupportedInstruction);
                }
            };
            let t = cx.operand_words(true_val, words)?;
            let f = cx.operand_words(false_val, words)?;
            let into = cx.result_words(result, words);
            let ty = cx.word();
            for (index, dst) in into.into_iter().enumerate() {
                cx.out.push(Instruction::Select {
                    ty,
                    dst,
                    cond: cond_reg.clone(),
                    true_val: t.at(index),
                    false_val: f.at(index),
                });
            }
        }
        IRInst::Load {
            result,
            ty,
            ptr,
            volatile: _,
            align,
        } if is_wide_type(ty, cx.int64_regs) => {
            reject_wide_alignment(*align, ty, "load")?;
            let at = cx.address_align(ptr);
            let base = address_base(ptr, cx)?;
            let words = wide_type_width(ty, cx.int64_regs)?;
            let loaded = cx.load_words_at(base, words, at);
            let into = cx.result_words(result, words);
            cx.write_words(into, loaded);
        }
        IRInst::Store {
            ty,
            ptr,
            value,
            volatile: _,
            align,
        } if is_wide_type(ty, cx.int64_regs) => {
            reject_wide_alignment(*align, ty, "store")?;
            let at = cx.address_align(ptr);
            let base = address_base(ptr, cx)?;
            let count = wide_type_width(ty, cx.int64_regs)?;
            let words = cx.operand_words(value, count)?;
            cx.store_words_at(base, words, at);
        }
        IRInst::Copy { result, ty, value } if is_wide_type(ty, cx.int64_regs) => {
            let count = wide_type_width(ty, cx.int64_regs)?;
            let words = cx.operand_words(value, count)?;
            let into = cx.result_words(result, count);
            cx.write_words(into, words);
        }
        IRInst::Ret { ty, value } if is_wide_type(ty, cx.int64_regs) => {
            let Some(value) = value else {
                return Err(FromIRError::InvalidIR);
            };
            let count = wide_type_width(ty, cx.int64_regs)?;
            let words = cx.operand_words(value, count)?;
            let address = cx.spill(words);
            cx.out.push(Instruction::Ret {
                value: Some(address),
            });
        }
        IRInst::Call {
            result,
            func_name,
            args,
        } => {
            let callee = cx.module.functions.get(func_name);
            let mut mir_args = Vec::with_capacity(args.len());
            for (index, (ty, arg)) in args.iter().enumerate() {
                let declared = callee.and_then(|f| f.signature.params.get(index));
                let no_bytes = match declared {
                    Some(param) => zero_sized(&param.ty, cx.module, cx.layout),
                    None => ty
                        .as_ref()
                        .is_some_and(|ty| zero_sized(ty, cx.module, cx.layout)),
                };
                if no_bytes {
                    continue;
                }
                let param_wide = declared.is_some_and(|p| is_wide_type(&p.ty, cx.int64_regs));
                let stated_wide = ty
                    .as_ref()
                    .is_some_and(|ty| is_wide_type(ty, cx.int64_regs));
                if param_wide || stated_wide || is_wide_value(arg, cx.vars, cx.int64_regs) {
                    let count = declared
                        .and_then(|p| type_words(&p.ty, cx.int64_regs))
                        .or_else(|| ty.as_ref().and_then(|ty| type_words(ty, cx.int64_regs)))
                        .or_else(|| value_words(arg, cx.vars, cx.int64_regs))
                        .ok_or(FromIRError::UnsupportedType)?;
                    let words = cx.operand_words(arg, count)?;
                    let address = cx.spill(words);
                    if declared.is_none() {
                        cx.mark_variadic_wide(&address, count, func_name)?;
                    }
                    mir_args.push(address);
                } else {
                    mir_args.push(resolve_operand(arg, cx.vregs, cx.vars, cx.globals)?);
                }
            }
            let returned = callee.and_then(|f| type_words(&f.signature.return_type, cx.int64_regs));
            if let Some(returned) = returned {
                let placeholder = cx.temp();
                cx.out.push(Instruction::Call {
                    name: (*func_name).to_string(),
                    args: mir_args,
                    ret: Some(placeholder.clone()),
                });
                if let Some(result) = result {
                    let loaded = cx.load_words(placeholder, returned);
                    let into = cx.result_words(result, returned);
                    cx.write_words(into, loaded);
                }
            } else {
                if callee.is_none() {
                    return Err(FromIRError::UnsupportedInstruction);
                }
                let zero_return = callee
                    .is_some_and(|f| zero_sized(&f.signature.return_type, cx.module, cx.layout));
                let ret = result
                    .filter(|_| !zero_return)
                    .map(|r| resolve_or_alloc(r, cx.vregs, cx.vars));
                cx.out.push(Instruction::Call {
                    name: (*func_name).to_string(),
                    args: mir_args,
                    ret,
                });
                if let Some(name) = result.filter(|_| zero_return) {
                    let dst = resolve_or_alloc(name, cx.vregs, cx.vars);
                    cx.out.push(zero_sized_address(dst, cx.vregs));
                }
            }
        }
        IRInst::CallIndirect {
            result,
            target,
            args,
            variadic_fixed,
            return_type,
        } => {
            // An aggregate reaches the callee as the bytes it stands for, named by the
            // classification the instruction carries. A legalized integer is spilled
            // first and then travels the same way. Past the fixed parameters the callee
            // reads an unnamed slot instead, which the backends place from what
            // `mark_variadic_wide` records.
            let passed: Vec<_> = args
                .iter()
                .filter(|(ty, _)| !zero_sized(ty, cx.module, cx.layout))
                .collect();
            let fixed = variadic_fixed.unwrap_or(passed.len());
            let mut mir_args = Vec::with_capacity(passed.len());
            let mut arg_types = Vec::with_capacity(passed.len());
            let mut arg_aggregates: Vec<Option<Aggregate>> = Vec::with_capacity(passed.len());
            for (index, (arg_ty, value)) in passed.into_iter().enumerate() {
                let wide = type_words(arg_ty, cx.int64_regs)
                    .or_else(|| value_words(value, cx.vars, cx.int64_regs));
                if let Some(count) = wide {
                    let words = cx.operand_words(value, count)?;
                    let address = cx.spill(words);
                    if index < fixed {
                        arg_aggregates.push(Some(words_aggregate(count, cx.int64_regs)));
                    } else {
                        cx.mark_variadic_wide(&address, count, "the called pointer")?;
                        arg_aggregates.push(None);
                    }
                    arg_types.push(MirType::Scalar(ScalarType::Ptr));
                    mir_args.push(address);
                    continue;
                }
                let aggregate = match resolve_type_names(arg_ty, &cx.module.type_declarations)
                    .filter(is_aggregate)
                {
                    Some(ty) => classify_aggregate(
                        &ty,
                        "argument of a call through a function pointer",
                        cx.layout,
                        cx.int64_regs,
                    )?,
                    None => None,
                };
                arg_aggregates.push(aggregate);
                let mut ty = match aggregate {
                    Some(_) => MirType::Scalar(ScalarType::Ptr),
                    None => map_ir_type(arg_ty, cx.layout)?,
                };
                let mut operand = resolve_operand(value, cx.vregs, cx.vars, cx.globals)?;
                normalize_lowered_literal(&mut operand, ty);
                // C promotes a float in the variadic tail to a double, as the direct
                // path does.
                if index >= fixed && ty == MirType::Scalar(ScalarType::F32) {
                    let dst = Register::Virtual(cx.vregs.allocate_fpr());
                    cx.out.push(Instruction::FloatConvert {
                        from: MirType::Scalar(ScalarType::F32),
                        to: MirType::Scalar(ScalarType::F64),
                        dst: dst.clone(),
                        src: operand,
                    });
                    operand = Operand::Register(dst);
                    ty = MirType::Scalar(ScalarType::F64);
                }
                arg_types.push(ty);
                mir_args.push(operand);
            }

            let zero_return = zero_sized(return_type, cx.module, cx.layout);
            let returned = type_words(return_type, cx.int64_regs).filter(|_| !zero_return);
            let composite = (!zero_return && returned.is_none())
                .then(|| resolve_type_names(return_type, &cx.module.type_declarations))
                .flatten()
                .filter(is_aggregate);
            let ret_aggregate = match (returned, &composite) {
                (Some(count), _) => Some(words_aggregate(count, cx.int64_regs)),
                (None, Some(ty)) => classify_aggregate(
                    ty,
                    "return of a call through a function pointer",
                    cx.layout,
                    cx.int64_regs,
                )?,
                (None, None) => None,
            };
            let ret_ty = match return_type {
                IRType::Void => None,
                _ if zero_return || ret_aggregate.is_some() => None,
                other => Some(map_ir_type(other, cx.layout)?),
            };
            let target = resolve_operand(target, cx.vregs, cx.vars, cx.globals)?;

            let mut after = None;
            let ret = match (ret_aggregate, result) {
                (Some(aggregate), _) => {
                    let slot = cx.vregs.allocate_gpr();
                    let words = aggregate.words();
                    if words > 1 {
                        cx.reservations.insert(slot, words - 1);
                    }
                    if let Some(name) = result {
                        after = Some((slot, returned, name));
                    }
                    Some(Register::Virtual(slot))
                }
                (None, Some(name)) if !zero_return => {
                    Some(resolve_or_alloc(name, cx.vregs, cx.vars))
                }
                (None, _) => None,
            };
            cx.out.push(Instruction::CallIndirect {
                target,
                arg_types,
                args: mir_args,
                variadic_fixed: *variadic_fixed,
                ret,
                ret_ty,
                ret_aggregate,
                arg_aggregates,
            });
            if let Some(name) = result.filter(|_| zero_return) {
                let dst = resolve_or_alloc(name, cx.vregs, cx.vars);
                cx.out.push(zero_sized_address(dst, cx.vregs));
            }
            if let Some((slot, returned, name)) = after {
                match returned {
                    Some(count) => {
                        let address = cx.temp();
                        cx.out.push(Instruction::StackAddr {
                            dst: address.clone(),
                            slot,
                        });
                        let words = cx.load_words(address, count);
                        let into = cx.result_words(name, count);
                        cx.write_words(into, words);
                    }
                    None => {
                        let dst = resolve_or_alloc(name, cx.vregs, cx.vars);
                        cx.out.push(Instruction::StackAddr { dst, slot });
                    }
                }
            }
        }
        IRInst::IntToFloat {
            result,
            signed,
            source_type,
            target_type,
            value,
        } if is_wide_prim(*source_type, cx.int64_regs) => {
            let count = wide_width(*source_type, cx.int64_regs)?;
            let words = cx.operand_words(value, count)?;
            let arg = cx.spill(words);
            let wide = count as i64 * cx.bits() == 128;
            let (symbol, ret_ty) = match target_type {
                IRPrim::F64 if *signed && wide => {
                    (RuntimeSymbol::Floattidf, MirType::Scalar(ScalarType::F64))
                }
                IRPrim::F64 if wide => {
                    (RuntimeSymbol::Floatuntidf, MirType::Scalar(ScalarType::F64))
                }
                IRPrim::F32 if *signed && wide => {
                    (RuntimeSymbol::Floattisf, MirType::Scalar(ScalarType::F32))
                }
                IRPrim::F32 if wide => {
                    (RuntimeSymbol::Floatuntisf, MirType::Scalar(ScalarType::F32))
                }
                IRPrim::F64 if *signed => {
                    (RuntimeSymbol::Floatdidf, MirType::Scalar(ScalarType::F64))
                }
                IRPrim::F64 => (RuntimeSymbol::Floatundidf, MirType::Scalar(ScalarType::F64)),
                IRPrim::F32 if *signed => {
                    (RuntimeSymbol::Floatdisf, MirType::Scalar(ScalarType::F32))
                }
                IRPrim::F32 => (RuntimeSymbol::Floatundisf, MirType::Scalar(ScalarType::F32)),
                IRPrim::I8
                | IRPrim::I16
                | IRPrim::I32
                | IRPrim::I64
                | IRPrim::U8
                | IRPrim::U16
                | IRPrim::U32
                | IRPrim::U64
                | IRPrim::I128
                | IRPrim::U128
                | IRPrim::Bool
                | IRPrim::Ptr => return Err(FromIRError::UnsupportedInstruction),
                IRPrim::F16 | IRPrim::BF16 => {
                    return Err(FromIRError::UnsupportedFloatFormat(
                        target_type.as_str().to_string(),
                    ));
                }
            };
            let name = cx.libcall_name(symbol, vec![Abi::Wide(count)], Some(Abi::Scalar(ret_ty)));
            let dst = resolve_or_alloc(result, cx.vregs, cx.vars);
            cx.out.push(Instruction::Call {
                name,
                args: vec![arg],
                ret: Some(dst),
            });
        }
        IRInst::FloatToInt {
            result,
            signed,
            source_type,
            target_type,
            value,
        } if is_wide_prim(*target_type, cx.int64_regs) => {
            let count = wide_width(*target_type, cx.int64_regs)?;
            let wide = count as i64 * cx.bits() == 128;
            let (symbol, param_ty) = match source_type {
                IRPrim::F64 if *signed && wide => {
                    (RuntimeSymbol::Fixdfti, MirType::Scalar(ScalarType::F64))
                }
                IRPrim::F64 if wide => {
                    (RuntimeSymbol::Fixunsdfti, MirType::Scalar(ScalarType::F64))
                }
                IRPrim::F32 if *signed && wide => {
                    (RuntimeSymbol::Fixsfti, MirType::Scalar(ScalarType::F32))
                }
                IRPrim::F32 if wide => {
                    (RuntimeSymbol::Fixunssfti, MirType::Scalar(ScalarType::F32))
                }
                IRPrim::F64 if *signed => {
                    (RuntimeSymbol::Fixdfdi, MirType::Scalar(ScalarType::F64))
                }
                IRPrim::F64 => (RuntimeSymbol::Fixunsdfdi, MirType::Scalar(ScalarType::F64)),
                IRPrim::F32 if *signed => {
                    (RuntimeSymbol::Fixsfdi, MirType::Scalar(ScalarType::F32))
                }
                IRPrim::F32 => (RuntimeSymbol::Fixunssfdi, MirType::Scalar(ScalarType::F32)),
                IRPrim::I8
                | IRPrim::I16
                | IRPrim::I32
                | IRPrim::I64
                | IRPrim::U8
                | IRPrim::U16
                | IRPrim::U32
                | IRPrim::U64
                | IRPrim::I128
                | IRPrim::U128
                | IRPrim::Bool
                | IRPrim::Ptr => return Err(FromIRError::UnsupportedInstruction),
                IRPrim::F16 | IRPrim::BF16 => {
                    return Err(FromIRError::UnsupportedFloatFormat(
                        target_type.as_str().to_string(),
                    ));
                }
            };
            let arg = resolve_operand(value, cx.vregs, cx.vars, cx.globals)?;
            let name = cx.libcall_name(symbol, vec![Abi::Scalar(param_ty)], Some(Abi::Wide(count)));
            let words = cx.call_wide_result(name, vec![arg], count);
            let into = cx.result_words(result, count);
            cx.write_words(into, words);
        }
        IRInst::Write {
            buffer,
            size,
            result,
        } => {
            let buffer = cx.narrow(buffer)?;
            let size = cx.narrow(size)?;
            io_call(cx, "lamina.write", vec![buffer, size], result);
        }
        IRInst::WriteByte { value, result } => {
            let value = cx.narrow(value)?;
            io_call(cx, "lamina.writebyte", vec![value], result);
        }
        IRInst::Read {
            buffer,
            size,
            result,
        } => {
            let buffer = cx.narrow(buffer)?;
            let size = cx.narrow(size)?;
            io_call(cx, "lamina.read", vec![buffer, size], result);
        }
        IRInst::ReadByte { result } => {
            io_call(cx, "lamina.readbyte", Vec::new(), result);
        }
        IRInst::Bitcast {
            result,
            source_type,
            target_type,
            value,
        } => {
            let from_pair = prim_words(*source_type, cx.int64_regs);
            let to_pair = prim_words(*target_type, cx.int64_regs);
            match (from_pair, to_pair) {
                (Some(from), Some(to)) => {
                    if from != to {
                        return Err(FromIRError::UnsupportedInstruction);
                    }
                    let words = cx.operand_words(value, from)?;
                    let into = cx.result_words(result, from);
                    cx.write_words(into, words);
                }
                (None, Some(count)) => {
                    let held =
                        map_ir_prim(*source_type, cx.layout)?.size_bytes(cx.layout.pointer_width);
                    if held != count * word_bytes(cx.int64_regs) as usize {
                        return Err(FromIRError::UnsupportedInstruction);
                    }
                    if let IRVal::Constant(literal) = value {
                        let parts = float_literal_words(literal, count, cx.int64_regs)?;
                        let words = Words(parts.into_iter().map(|word| cx.imm(word)).collect());
                        let into = cx.result_words(result, count);
                        cx.write_words(into, words);
                    } else {
                        let src = resolve_operand(value, cx.vregs, cx.vars, cx.globals)?;
                        let ty = map_ir_prim(*source_type, cx.layout)?;
                        let address = cx.words_slot(count);
                        cx.out.push(Instruction::Store {
                            ty,
                            src,
                            addr: AddressMode::BaseOffset {
                                base: address.clone(),
                                offset: 0,
                                address_space: Default::default(),
                            },
                            attrs: MemoryAttrs {
                                align: word_bytes(cx.int64_regs) as u8,
                                volatile: false,
                                stated_align: false,
                            },
                        });
                        let words = cx.load_words(address, count);
                        let into = cx.result_words(result, count);
                        cx.write_words(into, words);
                    }
                }
                (Some(count), None) => {
                    let words = cx.operand_words(value, count)?;
                    let address = cx.words_slot(count);
                    cx.store_words(address.clone(), words);
                    let ty = map_ir_prim(*target_type, cx.layout)?;
                    let dst = resolve_or_alloc(result, cx.vregs, cx.vars);
                    cx.out.push(Instruction::Load {
                        ty,
                        dst,
                        addr: AddressMode::BaseOffset {
                            base: address,
                            offset: 0,
                            address_space: Default::default(),
                        },
                        attrs: MemoryAttrs {
                            align: word_bytes(cx.int64_regs) as u8,
                            volatile: false,
                            stated_align: false,
                        },
                    });
                }
                (None, None) => return Err(FromIRError::UnsupportedInstruction),
            }
        }
        IRInst::PtrToInt {
            result,
            ptr_value,
            target_type,
        } if is_wide_prim(*target_type, cx.int64_regs) => {
            let src = resolve_operand(ptr_value, cx.vregs, cx.vars, cx.globals)?;
            let lo = cx.temp();
            let to = cx.word();
            cx.out.push(Instruction::IntConvert {
                from: MirType::Scalar(ScalarType::Ptr),
                to,
                signed: false,
                dst: lo.clone(),
                src,
            });
            let count = wide_width(*target_type, cx.int64_regs)?;
            let into = cx.result_words(result, count);
            let mut words = cx.zeros(count);
            words.0[0] = Operand::Register(lo);
            cx.write_words(into, words);
        }
        IRInst::IntToPtr {
            result,
            int_value,
            target_type,
        } if is_wide_prim(*target_type, cx.int64_regs) => {
            let src = cx.narrow(int_value)?;
            let from = cx.word();
            let dst = resolve_or_alloc(result, cx.vregs, cx.vars);
            cx.out.push(Instruction::IntConvert {
                from,
                to: MirType::Scalar(ScalarType::Ptr),
                signed: false,
                dst,
                src,
            });
        }
        IRInst::GetElemPtr {
            result,
            array_ptr,
            index,
            element_type,
        } => {
            let base = address_base(array_ptr, cx)?;
            let size = sizeof_type(element_type, cx.layout)
                .and_then(|size| i64::try_from(size).ok())
                .ok_or(FromIRError::UnsupportedInstruction)?;
            let index = cx.narrow(index)?;
            let scale = cx.imm(size);
            let scaled = cx.bin(IntBinOp::Mul, index, scale);
            let address = cx.temp();
            let word = cx.word();
            cx.out.push(Instruction::IntConvert {
                from: MirType::Scalar(ScalarType::Ptr),
                to: word,
                signed: false,
                dst: address.clone(),
                src: Operand::Register(base),
            });
            let sum = cx.bin(IntBinOp::Add, Operand::Register(address), scaled);
            let dst = resolve_or_alloc(result, cx.vregs, cx.vars);
            cx.out.push(Instruction::IntConvert {
                from: word,
                to: MirType::Scalar(ScalarType::Ptr),
                signed: false,
                dst,
                src: sum,
            });
        }
        IRInst::Binary { .. }
        | IRInst::Unary { .. }
        | IRInst::Cmp { .. }
        | IRInst::ZeroExtend { .. }
        | IRInst::SignExtend { .. }
        | IRInst::Trunc { .. }
        | IRInst::Load { .. }
        | IRInst::Store { .. }
        | IRInst::Copy { .. }
        | IRInst::Ret { .. }
        | IRInst::IntToFloat { .. }
        | IRInst::FloatToInt { .. }
        | IRInst::FloatConvert { .. }
        | IRInst::FuncAddr { .. }
        | IRInst::Switch { .. }
        | IRInst::Br { .. }
        | IRInst::Jmp { .. }
        | IRInst::Unreachable
        | IRInst::Alloc { .. }
        | IRInst::Fence { .. }
        | IRInst::GetFieldPtr { .. }
        | IRInst::PtrToInt { .. }
        | IRInst::IntToPtr { .. }
        | IRInst::VaStart { .. }
        | IRInst::VaEnd { .. }
        | IRInst::VaCopy { .. }
        | IRInst::InlineAsm { .. } => return Err(FromIRError::UnsupportedInstruction),
        IRInst::OverflowFlag {
            op,
            result,
            ty,
            lhs,
            rhs,
        } => {
            let words = wide_width(*ty, cx.int64_regs)?;
            let signed = !matches!(ty, IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64);
            let a = cx.operand_words(lhs, words)?;
            let b = cx.operand_words(rhs, words)?;
            let flag = cx.overflow(*op, signed, a, b);
            let dst = resolve_or_alloc(result, cx.vregs, cx.vars);
            let ty = cx.word();
            let zero = cx.imm(0);
            cx.out.push(Instruction::IntCmp {
                op: IntCmpOp::Ne,
                ty,
                dst,
                lhs: flag,
                rhs: zero,
            });
        }
        IRInst::VaArg { result, ty, ap } if is_wide_prim(*ty, cx.int64_regs) => {
            if cx.int64_regs {
                return Err(FromIRError::NotLegalizable(format!(
                    "va_arg.{ty}: no backend reads a 128-bit variadic argument, which the ABIs place as a pair of registers rather than one"
                )));
            }
            let count = wide_width(*ty, cx.int64_regs)?;
            let ap = resolve_operand(ap, cx.vregs, cx.vars, cx.globals)?;
            let slot = cx.vregs.allocate_gpr();
            cx.reservations
                .insert(slot, slot_reservation(count, cx.int64_regs));
            cx.vars
                .variadic_wide_results
                .insert(slot, words_aggregate(count, cx.int64_regs));
            cx.out.push(Instruction::VaArg {
                ty: map_ir_prim(*ty, cx.layout)?,
                dst: Register::Virtual(slot),
                ap,
            });
            let address = cx.temp();
            cx.out.push(Instruction::StackAddr {
                dst: address.clone(),
                slot,
            });
            let words = cx.load_words(address, count);
            let into = cx.result_words(result, count);
            cx.write_words(into, words);
        }
        IRInst::VaArg { .. } => return Ok(false),
        IRInst::AtomicLoad { ty, .. }
        | IRInst::AtomicStore { ty, .. }
        | IRInst::AtomicBinary { ty, .. }
        | IRInst::AtomicCompareExchange { ty, .. } => {
            return Err(FromIRError::NotLegalizable(format!(
                "`{instr}` is an atomic operation on {ty}, which takes two registers on a {}-bit target: two words cannot be read or written atomically without a double-word atomic",
                cx.bits()
            )));
        }
        IRInst::VectorBinary { .. }
        | IRInst::VectorSplat { .. }
        | IRInst::VectorExtractLane { .. }
        | IRInst::VectorInsertLane { .. }
        | IRInst::VectorShuffle { .. } => return Err(FromIRError::UnsupportedInstruction),
    }
    Ok(true)
}

/// An io intrinsic returns a signed count typed `i64`, so on a 32-bit target it defines
/// a pair: the call's own word, and that word's sign.
fn io_call<'a>(cx: &mut Cx<'_, 'a>, name: &str, args: Vec<Operand>, result: &'a str) {
    let into = cx.result_words(result, 2);
    cx.out.push(Instruction::Call {
        name: name.to_string(),
        args,
        ret: Some(into[0].clone()),
    });
    let top = cx.imm(cx.bits() - 1);
    let hi = cx.bin(IntBinOp::AShr, Operand::Register(into[0].clone()), top);
    let ty = cx.word();
    cx.out.push(Instruction::Copy {
        ty,
        dst: into[1].clone(),
        src: hi,
    });
}

/// The words of a float literal reinterpreted as an integer of the same width.
fn float_literal_words(
    literal: &Literal<'_>,
    words: usize,
    int64_regs: bool,
) -> Result<Vec<i64>, FromIRError> {
    let bits = match literal {
        Literal::F64(value) => value.to_bits() as i64,
        Literal::F32(value) => i64::from(value.to_bits()),
        other => return literal_words(other, words, int64_regs),
    };
    literal_words(&Literal::I64(bits), words, int64_regs)
}

/// Widens a value into a legalized integer of `words` registers. The source may itself
/// be legalized, in which case its words are kept and the rest take the fill.
fn widen<'a>(
    cx: &mut Cx<'_, 'a>,
    result: &'a str,
    source: IRPrim,
    value: &IRVal<'a>,
    signed: bool,
    words: usize,
) -> Result<(), FromIRError> {
    if let Some(held) = prim_words(source, cx.int64_regs) {
        if held >= words {
            return Err(FromIRError::UnsupportedInstruction);
        }
        let a = cx.operand_words(value, held)?;
        let fill = if signed {
            let top = cx.imm(cx.bits() - 1);
            let highest = a.top();
            cx.bin(IntBinOp::AShr, highest, top)
        } else {
            cx.imm(0)
        };
        let parts = (0..words)
            .map(|index| {
                if index < held {
                    a.at(index)
                } else {
                    fill.clone()
                }
            })
            .collect();
        let into = cx.result_words(result, words);
        cx.write_words(into, Words(parts));
        return Ok(());
    }
    let src = resolve_operand(value, cx.vregs, cx.vars, cx.globals)?;
    let word = cx.word_of(source, src, signed)?;
    let fill = if signed {
        let top = cx.imm(cx.bits() - 1);
        cx.bin(IntBinOp::AShr, word.clone(), top)
    } else {
        cx.imm(0)
    };
    let mut parts = vec![fill; words];
    parts[0] = word;
    let into = cx.result_words(result, words);
    cx.write_words(into, Words(parts));
    Ok(())
}

fn address_base<'a>(ptr: &IRVal<'a>, cx: &mut Cx<'_, 'a>) -> Result<Register, FromIRError> {
    match ptr {
        IRVal::Variable(id) => cx.vars.get(id).cloned().ok_or(FromIRError::UnknownVariable),
        IRVal::Global(name) => cx.globals.register(name, cx.vregs),
        IRVal::Constant(_) => Err(FromIRError::UnsupportedInstruction),
    }
}

pub(super) fn load_wide_parameter(
    address: Register,
    words: usize,
    vregs: &mut VirtualRegAllocator,
    prelude: &mut Vec<Instruction>,
    int64_regs: bool,
    endianness: Endianness,
) -> Vec<Register> {
    let word = word_ty(int64_regs);
    let attrs = MemoryAttrs {
        align: word_bytes(int64_regs) as u8,
        volatile: false,
        stated_align: false,
    };
    let mut read = |offset: i32| {
        let dst = Register::Virtual(vregs.allocate_gpr());
        let base = if offset == 0 {
            address.clone()
        } else {
            let stepped = Register::Virtual(vregs.allocate_gpr());
            prelude.push(Instruction::Lea {
                dst: stepped.clone(),
                base: address.clone(),
                offset,
            });
            stepped
        };
        prelude.push(Instruction::Load {
            ty: word,
            dst: dst.clone(),
            addr: AddressMode::BaseOffset {
                base,
                offset: 0,
                address_space: Default::default(),
            },
            attrs,
        });
        dst
    };
    (0..words)
        .map(|index| read(word_offset(index, words, int64_regs, endianness)))
        .collect()
}
