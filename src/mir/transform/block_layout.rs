//! Orders blocks so a loop body occupies one run of the index line.
//!
//! Live intervals are a single `[start, end]` pair over positions handed out by walking
//! `function.blocks` in storage order. A value defined in a loop and used later in the
//! same loop therefore covers every block stored between the two, whether or not the
//! loop can reach them. Inlining a call in a block that sits between them buries the
//! callee inside the interval, and the loop-carried values of the loop collide with
//! everything the callee needs. Reverse post-order puts a block before all the blocks it
//! dominates, so each loop lands in one run and an exit lands after it.
//!
//! Blocks the entry cannot reach keep their order and follow the rest.

use crate::mir::{
    Function,
    transform::{Transform, TransformCategory, TransformError, TransformLevel},
};
use std::collections::HashMap;

/// Reorders blocks into reverse post-order from the entry.
#[derive(Default)]
pub struct BlockLayout;

impl Transform for BlockLayout {
    fn name(&self) -> &'static str {
        "block_layout"
    }

    fn description(&self) -> &'static str {
        "Orders blocks in reverse post-order so each loop body is contiguous"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::ControlFlowOptimization
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Stable
    }

    fn apply(&self, func: &mut Function) -> Result<bool, TransformError> {
        Ok(reorder(func))
    }
}

/// The post-order of the blocks the entry reaches, by index into `func.blocks`.
fn postorder(func: &Function, index: &HashMap<&str, usize>) -> Vec<usize> {
    let Some(&entry) = index.get(func.entry.as_str()) else {
        return Vec::new();
    };
    let successors: Vec<Vec<usize>> = func
        .blocks
        .iter()
        .map(|block| {
            block
                .successors()
                .into_iter()
                .filter_map(|label| index.get(label).copied())
                .collect()
        })
        .collect();
    let mut order = Vec::with_capacity(func.blocks.len());
    let mut visited = vec![false; func.blocks.len()];
    let mut stack = vec![(entry, 0usize)];
    visited[entry] = true;
    while let Some((node, next)) = stack.last_mut() {
        let node = *node;
        if let Some(&succ) = successors[node].get(*next) {
            *next += 1;
            if !visited[succ] {
                visited[succ] = true;
                stack.push((succ, 0));
            }
        } else {
            order.push(node);
            stack.pop();
        }
    }
    order
}

fn reorder(func: &mut Function) -> bool {
    if func.blocks.len() < 2 {
        return false;
    }
    let index: HashMap<&str, usize> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(i, block)| (block.label.as_str(), i))
        .collect();
    if index.len() != func.blocks.len() {
        return false;
    }
    let mut order = postorder(func, &index);
    if order.is_empty() {
        return false;
    }
    order.reverse();
    let mut placed = vec![false; func.blocks.len()];
    for &i in &order {
        placed[i] = true;
    }
    order.extend((0..func.blocks.len()).filter(|&i| !placed[i]));
    if order.iter().copied().eq(0..func.blocks.len()) {
        return false;
    }
    let mut blocks: Vec<_> = func.blocks.drain(..).map(Some).collect();
    func.blocks = order
        .into_iter()
        .filter_map(|i| blocks[i].take())
        .collect();
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{Block, Instruction, Signature};
    use lamina_mir::Located;

    fn jmp(target: &str) -> Located {
        Located::new(Instruction::Jmp {
            target: target.to_string(),
        })
    }

    fn block(label: &str, terminator: Located) -> Block {
        Block {
            label: label.to_string(),
            instructions: vec![terminator],
        }
    }

    fn function(entry: &str, blocks: Vec<Block>) -> Function {
        let mut func = Function::new(Signature::new("f"));
        func.entry = entry.to_string();
        func.blocks = blocks;
        func
    }

    fn labels(func: &Function) -> Vec<&str> {
        func.blocks.iter().map(|b| b.label.as_str()).collect()
    }

    #[test]
    fn loop_body_becomes_contiguous() {
        // entry -> head -> {body, done}; body -> next -> head. `done` is stored between
        // the two halves of the body, which is the order inlining leaves behind.
        let mut func = function(
            "entry",
            vec![
                block("entry", jmp("head")),
                block(
                    "head",
                    Located::new(Instruction::Jmp {
                        target: "body".to_string(),
                    }),
                ),
                block("body", jmp("next")),
                block("done", jmp("done")),
                block("next", jmp("head")),
            ],
        );
        assert!(reorder(&mut func));
        let order = labels(&func);
        let body = order.iter().position(|l| *l == "body").unwrap();
        let next = order.iter().position(|l| *l == "next").unwrap();
        assert_eq!(next, body + 1, "loop body split by another block: {order:?}");
    }

    #[test]
    fn entry_stays_first() {
        let mut func = function(
            "entry",
            vec![
                block("other", jmp("entry")),
                block("entry", jmp("tail")),
                block("tail", jmp("tail")),
            ],
        );
        reorder(&mut func);
        assert_eq!(labels(&func)[0], "entry");
    }

    #[test]
    fn unreachable_blocks_are_kept() {
        let mut func = function(
            "entry",
            vec![
                block("entry", jmp("tail")),
                block("orphan", jmp("tail")),
                block("tail", jmp("tail")),
            ],
        );
        reorder(&mut func);
        let order = labels(&func);
        assert_eq!(order.len(), 3);
        assert!(order.contains(&"orphan"));
    }

    #[test]
    fn an_ordered_function_is_left_alone() {
        let mut func = function(
            "entry",
            vec![
                block("entry", jmp("mid")),
                block("mid", jmp("tail")),
                block("tail", jmp("tail")),
            ],
        );
        assert!(!reorder(&mut func));
        assert_eq!(labels(&func), ["entry", "mid", "tail"]);
    }
}
