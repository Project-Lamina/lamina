//! Basic block management for IR builder.
//!
//! This module provides methods for creating and managing basic blocks
//! within functions. Basic blocks are sequences of instructions with
//! a single entry and exit point.

use crate::builder::{BuildErrorKind, IRBuilder, PendingBlock, require};

impl<'a> IRBuilder<'a> {
    /// Opens a new basic block in the current function and makes it the current block.
    ///
    /// Panics when the label is already defined in the function.
    pub fn block(&mut self, name: &'a str) -> &mut Self {
        self.step(|builder| {
            let function = builder.open_body()?;
            require(
                function.blocks.iter().all(|block| block.name != name),
                BuildErrorKind::DuplicateBlock,
                || format!("block '{name}' is already defined in '@{}'", function.name),
            )?;
            function.blocks.push(PendingBlock {
                name,
                instructions: vec![],
            });
            let index = function.blocks.len() - 1;
            builder.current_block = Some(index);
            Ok(())
        })
    }

    /// Chooses which block of the current function runs first.
    ///
    /// Defaults to `entry`. The label must be defined by the time `build()` runs.
    pub fn set_entry_block(&mut self, name: &'a str) -> &mut Self {
        self.step(|builder| {
            builder.open_body()?.entry = Some(name);
            Ok(())
        })
    }
}
