//! Dead code elimination transform for MIR.

use crate::dense_set::DenseSet;
use crate::mir::{
    Block, Function, Instruction, Module, Register,
    transform::{Transform, TransformCategory, TransformError, TransformLevel},
};
use lamina_platform::PointerWidth;
use std::collections::{HashMap, HashSet};

/// Statistics about dead code elimination
#[derive(Debug, Default)]
pub struct DeadCodeStats {
    /// Number of instructions removed
    pub instructions_removed: usize,
    /// Number of registers that became dead
    pub registers_freed: usize,
}

/// Dead Code Elimination transform
#[derive(Default)]
pub struct DeadCodeElimination;

impl Transform for DeadCodeElimination {
    fn name(&self) -> &'static str {
        "dead_code_elimination"
    }

    fn description(&self) -> &'static str {
        "Removes instructions that define registers which are never used"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::DeadCodeElimination
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Stable
    }

    fn apply(&self, func: &mut Function) -> Result<bool, TransformError> {
        self.apply_internal(func)
            .map(|stats| stats.instructions_removed > 0)
    }
}

/// Dense numbering of the registers a function names, so liveness sets are bit sets.
struct RegisterIds {
    ids: HashMap<Register, usize>,
}

impl RegisterIds {
    fn new(func: &Function) -> Self {
        let mut ids: HashMap<Register, usize> = HashMap::new();
        for inst in func.blocks.iter().flat_map(|block| &block.instructions) {
            for reg in inst.def_reg().into_iter().chain(inst.use_regs()) {
                if !ids.contains_key(reg) {
                    let next = ids.len();
                    ids.insert(reg.clone(), next);
                }
            }
        }
        Self { ids }
    }

    fn width(&self) -> usize {
        self.ids.len()
    }

    fn id(&self, reg: &Register) -> Option<usize> {
        self.ids.get(reg).copied()
    }

    fn is_live(&self, live: &DenseSet, reg: &Register) -> bool {
        self.ids.get(reg).is_some_and(|&id| live.contains(id))
    }

    fn insert(&self, live: &mut DenseSet, reg: &Register) {
        if let Some(&id) = self.ids.get(reg) {
            live.insert(id);
        }
    }

    fn remove(&self, live: &mut DenseSet, reg: &Register) {
        if let Some(&id) = self.ids.get(reg) {
            live.remove(id);
        }
    }
}

/// What the liveness fixpoint needs from one instruction, with registers as dense ids.
struct LiveStep {
    def: Option<usize>,
    uses: Vec<usize>,
    /// Not a terminator and free of side effects, so it can be skipped when its result
    /// is not live.
    removable: bool,
}

impl DeadCodeElimination {
    /// Apply dead code elimination to a function
    pub fn apply_internal(&self, func: &mut Function) -> Result<DeadCodeStats, TransformError> {
        let mut stats = DeadCodeStats::default();
        let pointer = func.pointer_width;
        let ids = RegisterIds::new(func);

        // 1. Compute liveness analysis (inter-block)
        let live_out = self.compute_liveness(func, &ids)?;

        // 2. Remove dead instructions using liveness info
        for (block, live_at_end) in func.blocks.iter_mut().zip(live_out) {
            let mut live_regs = live_at_end;

            let removed =
                self.remove_dead_instructions_in_block(block, &mut live_regs, &ids, pointer);
            stats.instructions_removed += removed;
        }

        Ok(stats)
    }

    /// Compute liveness analysis: the registers live at the end of each block, in block
    /// order. Blocks that share a label share one entry, as they always have.
    fn compute_liveness(
        &self,
        func: &Function,
        ids: &RegisterIds,
    ) -> Result<Vec<DenseSet>, TransformError> {
        let pointer = func.pointer_width;
        let width = ids.width();
        let mut slot_of: HashMap<&str, usize> = HashMap::new();
        for block in &func.blocks {
            let next = slot_of.len();
            slot_of.entry(block.label.as_str()).or_insert(next);
        }
        let block_slot: Vec<usize> = func
            .blocks
            .iter()
            .map(|block| slot_of[block.label.as_str()])
            .collect();
        let steps: Vec<Vec<LiveStep>> = func
            .blocks
            .iter()
            .map(|block| {
                block
                    .instructions
                    .iter()
                    .map(|instr| LiveStep {
                        def: instr.def_reg().and_then(|reg| ids.id(reg)),
                        uses: instr
                            .use_regs()
                            .into_iter()
                            .filter_map(|reg| ids.id(reg))
                            .collect(),
                        removable: !instr.is_terminator()
                            && self.has_no_side_effects(instr, pointer),
                    })
                    .collect()
            })
            .collect();
        let mut live_in = vec![DenseSet::new(width); slot_of.len()];
        let mut live_out = vec![DenseSet::new(width); slot_of.len()];

        let mut changed = true;

        while changed {
            changed = false;

            // Process blocks in reverse order (heuristic for faster convergence)
            for (index, block) in func.blocks.iter().enumerate().rev() {
                let slot = block_slot[index];

                // 1. Calculate LiveOut = Union(LiveIn(successors))
                let mut current_live_out = DenseSet::new(width);
                if let Some(terminator) = block.instructions.last() {
                    let successors: Vec<&String> = match &terminator.instr {
                        Instruction::Jmp { target } => vec![target],
                        Instruction::Br {
                            true_target,
                            false_target,
                            ..
                        } => vec![true_target, false_target],
                        Instruction::Switch { cases, default, .. } => std::iter::once(default)
                            .chain(cases.iter().map(|(_, case_target)| case_target))
                            .collect(),
                        _ => Vec::new(), // Return or others have no successors within function
                    };
                    for target in successors {
                        if let Some(&succ) = slot_of.get(target.as_str()) {
                            current_live_out.union_with(&live_in[succ]);
                        }
                    }
                }

                if current_live_out != live_out[slot] {
                    live_out[slot] = current_live_out.clone();
                    changed = true;
                }

                // 2. Calculate LiveIn = Use U (LiveOut - Def)
                let mut current_live_in = current_live_out;
                for step in steps[index].iter().rev() {
                    if step.removable && step.def.is_none_or(|def| !current_live_in.contains(def)) {
                        continue;
                    }
                    if let Some(def) = step.def {
                        current_live_in.remove(def);
                    }
                    for &use_id in &step.uses {
                        current_live_in.insert(use_id);
                    }
                }

                if current_live_in != live_in[slot] {
                    live_in[slot] = current_live_in;
                    changed = true;
                }
            }
        }

        Ok(block_slot
            .into_iter()
            .map(|slot| live_out[slot].clone())
            .collect())
    }

    /// Remove dead instructions within a single basic block
    fn remove_dead_instructions_in_block(
        &self,
        block: &mut Block,
        live_regs: &mut DenseSet,
        ids: &RegisterIds,
        pointer: PointerWidth,
    ) -> usize {
        let mut dead = vec![false; block.instructions.len()];
        let mut removed_count = 0;
        for (index, instr) in block.instructions.iter().enumerate().rev() {
            if self.is_dead_instruction(instr, live_regs, ids, pointer) {
                dead[index] = true;
                removed_count += 1;
            } else {
                if let Some(def_reg) = instr.def_reg() {
                    ids.remove(live_regs, def_reg);
                }
                for use_reg in instr.use_regs() {
                    ids.insert(live_regs, use_reg);
                }
            }
        }

        if removed_count > 0 {
            let mut index = 0;
            block.instructions.retain(|_| {
                let keep = !dead[index];
                index += 1;
                keep
            });
        }

        removed_count
    }

    /// Check if an instruction is dead and can be safely removed
    fn is_dead_instruction(
        &self,
        instr: &Instruction,
        live_regs: &DenseSet,
        ids: &RegisterIds,
        pointer: PointerWidth,
    ) -> bool {
        // Never remove terminators
        if instr.is_terminator() {
            return false;
        }

        if let Some(def_reg) = instr.def_reg() {
            // If the defined register is not live, instruction is dead
            if !ids.is_live(live_regs, def_reg) {
                // Additional check: ensure instruction has no side effects
                return self.has_no_side_effects(instr, pointer);
            }
        } else {
            // Instructions that don't define a register are usually for side effects (e.g. stores)
            // But some might be useless?
            // Assuming has_no_side_effects check covers it.
            return self.has_no_side_effects(instr, pointer);
        }

        false
    }

    /// Check if an instruction has no side effects and can be safely removed
    fn has_no_side_effects(&self, instr: &Instruction, pointer: PointerWidth) -> bool {
        if instr.may_trap_integer(pointer) {
            return false;
        }
        match instr {
            // Pure arithmetic operations - safe to remove
            Instruction::Copy { .. }
            | Instruction::IntBinary { .. }
            | Instruction::FloatBinary { .. }
            | Instruction::FloatUnary { .. }
            | Instruction::IntUnary { .. }
            | Instruction::IntRotate { .. }
            | Instruction::IntCmp { .. }
            | Instruction::IntOverflow { .. }
            | Instruction::FloatCmp { .. }
            | Instruction::Select { .. }
            | Instruction::VectorOp { .. }
            | Instruction::IntConvert { .. }
            | Instruction::IntToFloat { .. }
            | Instruction::FloatToInt { .. }
            | Instruction::FloatConvert { .. }
            | Instruction::Bitcast { .. }
            | Instruction::StackAddr { .. }
            | Instruction::SymbolAddress { .. }
            | Instruction::ThreadLocalAddr { .. }
                | Instruction::Lea { .. } => true,

            // SIMD register-only ops are also pure (nightly only)

            // Instructions with side effects - never remove
            Instruction::Load { .. } // Reads memory (could fault)
            | Instruction::Store { .. }
            | Instruction::Call { .. }
            | Instruction::CallIndirect { .. }
            | Instruction::TailCall { .. }
            | Instruction::Ret { .. }
            | Instruction::Jmp { .. }
            | Instruction::Br { .. }
            | Instruction::Switch { .. }
            | Instruction::Unreachable
            | Instruction::SafePoint
            | Instruction::StackMap { .. }
            | Instruction::PatchPoint { .. }
            | Instruction::Comment { .. } => false,

            // Atomic operations and fences have side effects
            Instruction::AtomicLoad { .. }
            | Instruction::AtomicStore { .. }
            | Instruction::AtomicBinary { .. }
            | Instruction::AtomicCompareExchange { .. }
            | Instruction::Fence { .. } => false,

            Instruction::InlineAsm { .. } => false,

            // Variadic argument access writes through its va_list
            Instruction::VaStart { .. }
            | Instruction::VaArg { .. }
            | Instruction::VaEnd { .. }
            | Instruction::VaCopy { .. } => false,

            // SIMD memory operations have side effects (nightly only)
        }
    }
}

/// Ends each block at the first call to a `@noreturn` function, so the code after it is
/// unreachable. Returns the number of blocks changed.
pub fn truncate_after_noreturn_calls(module: &mut Module) -> usize {
    let noreturn: HashSet<String> = module
        .functions
        .iter()
        .filter(|(_, func)| func.sig.noreturn)
        .map(|(name, _)| name.clone())
        .collect();
    if noreturn.is_empty() {
        return 0;
    }

    let mut changed = 0;
    for func in module.functions.values_mut() {
        for block in &mut func.blocks {
            let Some(index) = block.instructions.iter().position(
                |inst| matches!(&inst.instr, Instruction::Call { name, .. } if noreturn.contains(name)),
            ) else {
                continue;
            };
            let already_ends = block.instructions.len() == index + 2
                && matches!(
                    &block.instructions.last().map(|i| &i.instr),
                    Some(Instruction::Unreachable)
                );
            if already_ends {
                continue;
            }
            block.instructions.truncate(index + 1);
            block.instructions.push(Instruction::Unreachable.into());
            changed += 1;
        }
    }
    changed
}

/// Deletes calls to `@pure` or `@const` functions whose result is never read. Repeats
/// until a round removes nothing, since removing a call can leave its argument unread.
/// Returns the number of calls removed.
pub fn remove_dead_pure_calls(module: &mut Module) -> usize {
    let pure: HashSet<String> = module
        .functions
        .iter()
        .filter(|(_, func)| func.sig.pure)
        .map(|(name, _)| name.clone())
        .collect();
    if pure.is_empty() {
        return 0;
    }

    let mut total = 0;
    loop {
        let mut round = 0;
        for func in module.functions.values_mut() {
            let mut uses: HashMap<Register, usize> = HashMap::new();
            for inst in func.blocks.iter().flat_map(|b| &b.instructions) {
                for reg in inst.use_regs() {
                    *uses.entry(reg.clone()).or_insert(0) += 1;
                }
            }
            for block in &mut func.blocks {
                let before = block.instructions.len();
                block.instructions.retain(|inst| match &inst.instr {
                    Instruction::Call { name, ret, .. } if pure.contains(name) => match ret {
                        Some(reg) => uses.get(reg).is_some_and(|&n| n > 0),
                        None => false,
                    },
                    _ => true,
                });
                round += before - block.instructions.len();
            }
        }
        if round == 0 {
            break;
        }
        total += round;
    }
    total
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mir::{
        AddressMode, FunctionBuilder, Immediate, IntBinOp, MemoryAttrs, MirType, Operand,
        ScalarType, VirtualReg, transform::test_utils::get_block,
    };
    use lamina_mir::Located;

    fn defines(block: &Block, reg: VirtualReg) -> bool {
        let reg: Register = reg.into();
        block.instructions.iter().any(|i| i.def_reg() == Some(&reg))
    }

    fn ends_with_ret(block: &Block) -> bool {
        matches!(
            &block.instructions.last().map(|i| &i.instr),
            Some(Instruction::Ret { .. })
        )
    }

    fn has_int_binary(block: &Block) -> bool {
        block
            .instructions
            .iter()
            .any(|i| matches!(&i.instr, Instruction::IntBinary { .. }))
    }

    #[test]
    fn test_dead_code_elimination_basic() {
        // Create a function with dead instructions
        let func = FunctionBuilder::new("test")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            // Dead instruction: v1 = v0 + 42 (v1 never used)
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(42)),
            })
            // Live instruction: v2 = v0 + 10 (v2 used in ret)
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(2).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(10)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(2).into())),
            })
            .build();

        let mut func = func;
        let dce = DeadCodeElimination;

        let changed = dce.apply(&mut func).expect("DCE should succeed");

        assert!(changed);

        let entry = func.get_block("entry").expect("entry block exists");
        assert!(!defines(entry, VirtualReg::gpr(1)));
        assert!(defines(entry, VirtualReg::gpr(2)));
        assert!(ends_with_ret(entry));
    }

    #[test]
    fn test_dce_empty_function() {
        // Test with empty function - should not panic
        let mut func = FunctionBuilder::new("empty")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Ret { value: None })
            .build();

        let dce = DeadCodeElimination;
        let result = dce.apply(&mut func);
        assert!(result.is_ok());
        assert!(!result.unwrap()); // No changes expected
    }

    #[test]
    fn test_dce_single_block_all_live() {
        // All instructions are live - nothing should be removed
        let mut func = FunctionBuilder::new("all_live")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(1).into())),
            })
            .build();

        let dce = DeadCodeElimination;
        let changed = dce.apply(&mut func).expect("should succeed");
        assert!(!changed);
        assert!(defines(&func.blocks[0], VirtualReg::gpr(1)));
        assert!(ends_with_ret(&func.blocks[0]));
    }

    #[test]
    fn test_dce_preserves_terminators() {
        // Terminators must never be removed even if they define no live registers
        let mut func = FunctionBuilder::new("terminators")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Jmp {
                target: "exit".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret { value: None })
            .build();

        let dce = DeadCodeElimination;
        let result = dce.apply(&mut func);
        assert!(result.is_ok());

        // Jmp and Ret should still exist
        let entry = get_block(&func, "entry");
        assert!(matches!(
            &entry.instructions.last().map(|i| &i.instr),
            Some(Instruction::Jmp { .. })
        ));
    }

    #[test]
    fn test_dce_preserves_side_effects() {
        // Store and Call have side effects - must not be removed
        let mut func = FunctionBuilder::new("side_effects")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Store {
                ty: MirType::Scalar(ScalarType::I64),
                src: Operand::Immediate(Immediate::I64(42)),
                addr: AddressMode::BaseOffset {
                    base: VirtualReg::gpr(0).into(),
                    offset: 0,
                    address_space: Default::default(),
                },
                attrs: MemoryAttrs::default(),
            })
            .instr(Instruction::Call {
                name: "lamina.writebyte".to_string(),
                args: vec![Operand::Immediate(Immediate::I64(1))],
                ret: None,
                ret_ty: None,
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Immediate(Immediate::I64(0))),
            })
            .build();

        let dce = DeadCodeElimination;
        let changed = dce.apply(&mut func).expect("should succeed");

        assert!(!changed);
        let entry = &func.blocks[0];
        assert!(matches!(
            &entry.instructions[0].instr,
            Instruction::Store { .. }
        ));
        assert!(matches!(
            &entry.instructions[1].instr,
            Instruction::Call { .. }
        ));
        assert!(ends_with_ret(entry));
    }

    #[test]
    fn test_dce_chain_of_dead_instructions() {
        // Chain of dead instructions: v1 = f(v0), v2 = f(v1), v3 = f(v2)
        // None used in return
        let mut func = FunctionBuilder::new("dead_chain")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(2).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Immediate(Immediate::I64(2)),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(3).into(),
                lhs: Operand::Register(VirtualReg::gpr(2).into()),
                rhs: Operand::Immediate(Immediate::I64(3)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Immediate(Immediate::I64(99))),
            })
            .build();

        let dce = DeadCodeElimination;
        let changed = dce.apply(&mut func).expect("should succeed");
        assert!(changed);
        let entry = &func.blocks[0];
        assert!(!has_int_binary(entry));
        assert!(!defines(entry, VirtualReg::gpr(1)));
        assert!(!defines(entry, VirtualReg::gpr(2)));
        assert!(!defines(entry, VirtualReg::gpr(3)));
        assert!(ends_with_ret(entry));
    }

    #[test]
    fn test_dce_multi_block_liveness() {
        // Value defined in entry, used in exit block
        let mut func = FunctionBuilder::new("multi_block")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(10)),
            })
            .instr(Instruction::Jmp {
                target: "exit".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(1).into())),
            })
            .build();

        let dce = DeadCodeElimination;
        let changed = dce.apply(&mut func).expect("should succeed");

        // v1 is used in exit block, so entry's add should remain
        assert!(!changed);
        let entry = get_block(&func, "entry");
        assert!(defines(entry, VirtualReg::gpr(1)));
        assert!(matches!(
            &entry.instructions.last().map(|i| &i.instr),
            Some(Instruction::Jmp { .. })
        ));
    }

    #[test]
    fn test_dce_loop_back_edge_liveness() {
        // Test liveness across loop back-edge
        // v0 defined in entry, used in loop, loop branches back
        let mut func = FunctionBuilder::new("loop_liveness")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Jmp {
                target: "loop".to_string(),
            })
            .block("loop")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(0).into(),
                true_target: "loop".to_string(),
                false_target: "exit".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(0).into())),
            })
            .build();

        let dce = DeadCodeElimination;
        let result = dce.apply(&mut func);
        assert!(result.is_ok());

        // Loop body should be preserved (v0 is used)
        let loop_block = get_block(&func, "loop");
        assert!(defines(loop_block, VirtualReg::gpr(0)));
        assert!(matches!(
            &loop_block.instructions.last().map(|i| &i.instr),
            Some(Instruction::Br { .. })
        ));
    }

    #[test]
    fn test_dce_does_not_infinite_loop() {
        // Stress test: large function should not cause infinite loop
        let mut func = FunctionBuilder::new("stress")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .build();

        // Add 500 dead instructions
        for i in 0..500 {
            func.blocks[0].instructions.insert(
                0,
                Located::from(Instruction::IntBinary {
                    op: IntBinOp::Add,
                    ty: MirType::Scalar(ScalarType::I64),
                    dst: VirtualReg::gpr(i + 1).into(),
                    lhs: Operand::Immediate(Immediate::I64(i as i64)),
                    rhs: Operand::Immediate(Immediate::I64(1)),
                }),
            );
        }
        func.blocks[0]
            .instructions
            .push(Located::from(Instruction::Ret {
                value: Some(Operand::Immediate(Immediate::I64(0))),
            }));

        let dce = DeadCodeElimination;
        let result = dce.apply(&mut func);
        assert!(result.is_ok());
        assert!(!has_int_binary(&func.blocks[0]));
        assert!(ends_with_ret(&func.blocks[0]));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod annotation_tests {
    use super::*;
    use crate::{
        mir::{codegen::from_ir_x86_64, transform::TransformPipeline},
        mir_codegen::x86_64::generate_mir_x86_64,
        parser::parse_module,
    };
    use lamina_platform::TargetOperatingSystem;

    fn lower(source: &str) -> Module {
        let ir = parse_module(source).expect("parse");
        from_ir_x86_64(&ir, "annotated").expect("lower")
    }

    fn calls_in(func: &Function) -> usize {
        func.blocks
            .iter()
            .flat_map(|b| &b.instructions)
            .filter(|i| matches!(&i.instr, Instruction::Call { .. }))
            .count()
    }

    const NORETURN_CALLER: &str = "fn @stop() -> void {\nentry:\n ret.void\n}\nfn @main() -> i64 {\nentry:\n call @stop()\n %x = add.i64 1, 2\n ret.i64 %x\n}";

    #[test]
    fn code_after_noreturn_call_becomes_unreachable() {
        let source = NORETURN_CALLER.replacen("fn @stop", "@noreturn fn @stop", 1);
        let mut module = lower(&source);
        assert_eq!(truncate_after_noreturn_calls(&mut module), 1);
        let block = &module.get_function("main").expect("main").blocks[0];
        assert_eq!(block.instructions.len(), 2, "{:?}", block.instructions);
        assert!(matches!(
            &block.instructions[0].instr,
            Instruction::Call { .. }
        ));
        assert!(matches!(
            &block.instructions[1].instr,
            Instruction::Unreachable
        ));
    }

    #[test]
    fn calls_to_returning_functions_keep_following_code() {
        let mut module = lower(NORETURN_CALLER);
        assert_eq!(truncate_after_noreturn_calls(&mut module), 0);
        let block = &module.get_function("main").expect("main").blocks[0];
        assert!(
            !block
                .instructions
                .iter()
                .any(|i| matches!(&i.instr, Instruction::Unreachable))
        );
        assert!(matches!(
            &block.instructions.last().map(|i| &i.instr),
            Some(Instruction::Ret { .. })
        ));
    }

    #[test]
    fn noreturn_truncation_survives_the_standard_pipeline() {
        let source = NORETURN_CALLER.replacen("fn @stop", "@noreturn fn @stop", 1);
        let mut module = lower(&source);
        TransformPipeline::default_for_opt_level(1)
            .apply_to_module(&mut module)
            .expect("optimize");
        let main = module.get_function("main").expect("main");
        let instructions: Vec<&Instruction> = main
            .blocks
            .iter()
            .flat_map(|b| &b.instructions)
            .map(|i| &i.instr)
            .collect();
        assert!(
            instructions
                .iter()
                .any(|i| matches!(i, Instruction::Call { .. }))
        );
        assert!(matches!(
            instructions.last(),
            Some(Instruction::Unreachable)
        ));
        assert!(
            !instructions
                .iter()
                .any(|i| matches!(i, Instruction::IntBinary { .. }))
        );
    }

    #[test]
    fn noreturn_call_emits_trap_after_the_call_on_x86_64() {
        let source = NORETURN_CALLER.replacen("fn @stop", "@noreturn fn @stop", 1);
        let mut module = lower(&source);
        TransformPipeline::default_for_opt_level(1)
            .apply_to_module(&mut module)
            .expect("optimize");
        let mut output = Vec::new();
        generate_mir_x86_64(&module, &mut output, TargetOperatingSystem::Linux)
            .expect("emit x86_64");
        let text = String::from_utf8(output).expect("utf8 assembly");
        let call = text.find("call stop").expect(&text);
        assert!(text[call..].contains("ud2"), "{text}");
    }

    #[test]
    fn noreturn_call_compiles_on_every_native_backend() {
        use crate::error::LaminaError;
        use crate::mir::Module;
        use crate::mir_codegen::{
            arm::aarch64::generate_mir_aarch64, powerpc::powerpc64::generate_mir_ppc64,
            riscv::generate_mir_riscv,
        };
        let source = NORETURN_CALLER.replacen("fn @stop", "@noreturn fn @stop", 1);
        let mut module = lower(&source);
        TransformPipeline::default_for_opt_level(1)
            .apply_to_module(&mut module)
            .expect("optimize");
        type Emitter = fn(&Module, &mut Vec<u8>, TargetOperatingSystem) -> Result<(), LaminaError>;
        let emitters: [(&str, Emitter); 4] = [
            ("x86_64", generate_mir_x86_64::<Vec<u8>>),
            ("aarch64", |module, writer, os| {
                generate_mir_aarch64(module, writer, os, PointerWidth::Bits64)
            }),
            ("riscv", generate_mir_riscv::<Vec<u8>>),
            ("ppc64le", generate_mir_ppc64::<Vec<u8>>),
        ];
        for (name, emit) in emitters {
            let mut output = Vec::new();
            emit(&module, &mut output, TargetOperatingSystem::Linux)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }

    fn pure_source(annotation: &str, used: bool) -> String {
        let tail = if used {
            "%v = call @k()\n ret.i64 %v"
        } else {
            "call @k()\n ret.i64 1"
        };
        format!(
            "{annotation}fn @k() -> i64 {{\nentry:\n ret.i64 4\n}}\nfn @main() -> i64 {{\nentry:\n {tail}\n}}"
        )
    }

    #[test]
    fn unused_pure_call_is_removed() {
        let mut module = lower(&pure_source("@pure ", false));
        assert_eq!(remove_dead_pure_calls(&mut module), 1);
        assert_eq!(calls_in(module.get_function("main").expect("main")), 0);
    }

    #[test]
    fn unused_call_to_unannotated_function_is_kept() {
        let mut module = lower(&pure_source("", false));
        assert_eq!(remove_dead_pure_calls(&mut module), 0);
        assert_eq!(calls_in(module.get_function("main").expect("main")), 1);
    }

    #[test]
    fn used_pure_call_is_kept() {
        let mut module = lower(&pure_source("@pure ", true));
        assert_eq!(remove_dead_pure_calls(&mut module), 0);
        assert_eq!(calls_in(module.get_function("main").expect("main")), 1);
    }
}
