//! Recomputes a symbol's address in each block that reads it.
//!
//! An address of a global or function is a single cheap instruction, so a block that reads
//! one defined elsewhere gets its own copy instead of keeping it live across the function.
//! Otherwise a value used only on a cold path sits in a stack slot for the whole loop.

use crate::mir::{
    Function, Instruction, Register, VirtualReg,
    transform::{
        Transform, TransformCategory, TransformError, TransformLevel, inline::ModuleInlining,
        loop_rotation::next_virtual_id,
    },
};
use lamina_mir::Located;
use std::collections::HashMap;

/// Gives each block its own copy of the symbol addresses it reads.
#[derive(Default)]
pub struct RematerializeSymbols;

impl Transform for RematerializeSymbols {
    fn name(&self) -> &'static str {
        "rematerialize_symbols"
    }

    fn description(&self) -> &'static str {
        "Recomputes symbol addresses in each block that reads them"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::CopyPropagation
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Experimental
    }

    fn apply(&self, func: &mut Function) -> Result<bool, TransformError> {
        Ok(rematerialize(func))
    }
}

fn rematerialize(func: &mut Function) -> bool {
    let mut defined_in: HashMap<Register, (usize, String)> = HashMap::new();
    for (index, block) in func.blocks.iter().enumerate() {
        for inst in &block.instructions {
            if let Instruction::SymbolAddress { dst, symbol } = &inst.instr {
                defined_in.insert(dst.clone(), (index, symbol.clone()));
            }
        }
    }
    if defined_in.is_empty() {
        return false;
    }

    let mut next_id = next_virtual_id(func);
    let mut changed = false;
    for block_index in 0..func.blocks.len() {
        let mut copies: HashMap<Register, Register> = HashMap::new();
        let mut inserted: Vec<Located> = Vec::new();
        let read: Vec<Register> = func.blocks[block_index]
            .instructions
            .iter()
            .flat_map(|inst| {
                inst.instr
                    .use_regs()
                    .into_iter()
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .collect();
        for reg in read {
            let Some((def_block, symbol)) = defined_in.get(&reg) else {
                continue;
            };
            if *def_block == block_index || copies.contains_key(&reg) {
                continue;
            }
            let fresh = Register::Virtual(VirtualReg::gpr(next_id));
            next_id += 1;
            inserted.push(
                Instruction::SymbolAddress {
                    dst: fresh.clone(),
                    symbol: symbol.clone(),
                }
                .into(),
            );
            copies.insert(reg, fresh);
        }
        if copies.is_empty() {
            continue;
        }
        changed = true;
        let block = &mut func.blocks[block_index];
        for inst in &mut block.instructions {
            ModuleInlining::map_registers(&mut inst.instr, &mut |reg: &Register| {
                copies.get(reg).cloned().unwrap_or_else(|| reg.clone())
            });
        }
        block.instructions.splice(0..0, inserted);
    }
    changed
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};

    fn lowered(source: &str) -> Function {
        let ir = parse_module(source).expect("parse");
        let module = from_ir_x86_64(&ir, "remat").expect("lower");
        module.functions.get("main").expect("main").clone()
    }

    fn symbol_defs(func: &Function) -> Vec<(usize, String)> {
        func.blocks
            .iter()
            .enumerate()
            .flat_map(|(b, block)| {
                block
                    .instructions
                    .iter()
                    .filter_map(move |inst| match &inst.instr {
                        Instruction::SymbolAddress { symbol, .. } => Some((b, symbol.clone())),
                        _ => None,
                    })
            })
            .collect()
    }

    #[test]
    fn a_symbol_read_in_another_block_is_addressed_there_too() {
        let source = "global $msg: [3 x i8] = \"hi\"\nfn @main(i64 %a) -> i64 {\nentry:\n    %e = load.i64 $msg\n    %c = eq.i64 %a, 0\n    br %c, cold, hot\ncold:\n    %w = load.i64 $msg\n    ret.i64 %w\nhot:\n    ret.i64 %a\n}\n";
        let mut func = lowered(source);
        assert!(rematerialize(&mut func));
        let defs = symbol_defs(&func);
        assert_eq!(defs.len(), 2, "{defs:?}");
        assert!(defs.iter().all(|(_, symbol)| symbol == "msg"), "{defs:?}");
        assert!(defs.iter().any(|(b, _)| *b == 1), "{defs:?}");
    }

    #[test]
    fn a_function_without_symbol_reads_is_unchanged() {
        let mut func = lowered("fn @main(i64 %a) -> i64 {\nentry:\n    ret.i64 %a\n}\n");
        assert!(!rematerialize(&mut func));
    }
}
