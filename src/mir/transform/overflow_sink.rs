//! Moves a checked operation down next to the branch that reads its flag.

use crate::mir::{
    Block, Function, Instruction, Register, VirtualReg,
    transform::{Transform, TransformCategory, TransformError, TransformLevel},
};
use std::collections::HashMap;

/// Sinks checked operations next to the branch that tests them.
#[derive(Default)]
pub struct SinkOverflowChecks;

impl Transform for SinkOverflowChecks {
    fn name(&self) -> &'static str {
        "sink_overflow_checks"
    }

    fn description(&self) -> &'static str {
        "Moves checked operations next to the branch that reads their overflow flag"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::ControlFlowOptimization
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Experimental
    }

    fn apply(&self, func: &mut Function) -> Result<bool, TransformError> {
        let uses = use_counts(func);
        let mut changed = false;
        for block in &mut func.blocks {
            changed |= sink_into_branch(block, &uses);
        }
        Ok(changed)
    }
}

fn use_counts(func: &Function) -> HashMap<VirtualReg, usize> {
    let mut counts = HashMap::new();
    for inst in func.blocks.iter().flat_map(|block| &block.instructions) {
        for reg in inst.use_regs() {
            if let Register::Virtual(reg) = reg {
                *counts.entry(*reg).or_insert(0) += 1;
            }
        }
    }
    counts
}

fn sink_into_branch(block: &mut Block, uses: &HashMap<VirtualReg, usize>) -> bool {
    let len = block.instructions.len();
    let Some(Instruction::Br {
        cond: Register::Virtual(cond),
        ..
    }) = block.instructions.last().map(|inst| &inst.instr)
    else {
        return false;
    };
    if uses.get(cond) != Some(&1) {
        return false;
    }
    let Some(pos) = block.instructions.iter().position(|inst| {
        matches!(
            &inst.instr,
            Instruction::IntOverflow { dst: Register::Virtual(dst), .. } if dst == cond
        )
    }) else {
        return false;
    };
    if pos + 2 >= len {
        return false;
    }
    let operands: Vec<Register> = block.instructions[pos]
        .instr
        .use_regs()
        .into_iter()
        .cloned()
        .collect();
    let redefined = block.instructions[pos + 1..len - 1]
        .iter()
        .filter_map(|inst| inst.instr.def_reg())
        .any(|def| operands.contains(def));
    if redefined {
        return false;
    }
    let moved = block.instructions.remove(pos);
    block.instructions.insert(len - 2, moved);
    true
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};

    fn sunk(source: &str) -> (bool, Vec<String>) {
        let ir = parse_module(source).expect("parse");
        let mut module = from_ir_x86_64(&ir, "sink").expect("lower");
        let func = module.functions.get_mut("main").expect("main");
        let changed = SinkOverflowChecks.apply(func).expect("sink");
        let names = func
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .map(|inst| format!("{}", inst.instr))
            .collect();
        (changed, names)
    }

    fn position_of(names: &[String], needle: &str) -> usize {
        names
            .iter()
            .position(|name| name.contains(needle))
            .unwrap_or_else(|| panic!("no `{needle}` in {names:?}"))
    }

    #[test]
    fn checked_add_moves_next_to_its_branch() {
        let (changed, names) = sunk(
            "fn @main(i64 %a, i64 %b) -> i64 {\nentry:\n    %f = add.ovf.u64 %a, %b\n    %w = add.i64 %a, 1\n    br %f, fail, ok\nok:\n    ret.i64 %w\nfail:\n    ret.i64 0\n}\n",
        );
        assert!(changed);
        let overflow = position_of(&names, "add.ovf");
        let branch = position_of(&names, "br ");
        assert_eq!(overflow + 1, branch, "{names:?}");
    }

    #[test]
    fn checked_op_stays_when_an_operand_is_redefined_before_the_branch() {
        let (changed, names) = sunk(
            "fn @main(i64 %a, i64 %b) -> i64 {\nentry:\n    %f = add.ovf.u64 %a, %b\n    %a = add.i64 %a, 1\n    br %f, fail, ok\nok:\n    ret.i64 %a\nfail:\n    ret.i64 0\n}\n",
        );
        assert!(!changed);
        let overflow = position_of(&names, "add.ovf");
        let redefinition = position_of(&names, "add.i64");
        assert!(overflow < redefinition, "{names:?}");
    }

    #[test]
    fn flag_read_by_more_than_the_branch_stays_put() {
        let (changed, _) = sunk(
            "fn @main(i64 %a, i64 %b) -> i64 {\nentry:\n    %f = add.ovf.u64 %a, %b\n    %w = extend.bool.i64 %f\n    %x = add.i64 %a, 1\n    br %f, fail, ok\nok:\n    ret.i64 %x\nfail:\n    ret.i64 %w\n}\n",
        );
        assert!(!changed);
    }
}
