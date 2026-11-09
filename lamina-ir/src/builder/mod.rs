//! # IR Builder
//!
//! A fluent API for programmatically constructing Lamina IR modules.

mod annotations;
mod arithmetic;
mod atomics;
mod blocks;
pub(crate) mod checks;
mod control_flow;
mod conversions;
mod functions;
mod globals;
mod io;
mod memory;
mod pointers;
mod simd;
mod types;
mod values;

#[cfg(test)]
mod tests;

use std::{collections::HashMap, error::Error, fmt};

use crate::{
    function::{BasicBlock, Function, FunctionAnnotation, FunctionSignature},
    instruction::Instruction,
    module::Module,
};

// Re-export value factory functions
pub use values::*;

/// # IR Builder
///
/// A fluent API for programmatically constructing Lamina IR modules.
///
/// ## Overview
///
/// The `IRBuilder` allows you to construct IR code in a safe, programmatic way without
/// having to manually build instruction objects. It provides methods for all IR operations
/// and maintains the context of the current function and basic block.
///
/// Every method emits IR that the text parser accepts. Misuse that would produce IR the
/// parser rejects, or a different meaning than requested, panics with the replacement to use.
/// A builder made with `IRBuilder::new_fallible()` records the first such misuse instead,
/// ignores the calls after it, and reports the error from `try_build()`.
///
/// ## Basic Usage Pattern
///
/// 1. Create a builder with `IRBuilder::new()`
/// 2. Define a function with `function()` or `function_with_params()`
/// 3. Add basic blocks with `block()`
/// 4. Add instructions to the current block
/// 5. Generate the final module with `build()`
pub struct IRBuilder<'a> {
    pub(super) module: Module<'a>,
    pub(super) functions: Vec<PendingFunction<'a>>,
    pub(super) current_function: Option<usize>,
    pub(super) current_block: Option<usize>,
    pub(super) temp_var_counter: usize,
    pub(super) fallible: bool,
    pub(super) error: Option<BuildError>,
}

/// A function under construction, kept in definition order.
pub(super) struct PendingFunction<'a> {
    pub(super) name: &'a str,
    pub(super) signature: FunctionSignature<'a>,
    pub(super) annotations: Vec<FunctionAnnotation>,
    pub(super) is_extern: bool,
    pub(super) blocks: Vec<PendingBlock<'a>>,
    pub(super) entry: Option<&'a str>,
}

/// A basic block under construction. Labels are scoped to their function.
pub(super) struct PendingBlock<'a> {
    pub(super) name: &'a str,
    pub(super) instructions: Vec<Instruction<'a>>,
}

/// Misuse of the builder, or a module it cannot finish.
///
/// Its message is the text the panicking methods panic with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildError {
    kind: BuildErrorKind,
    message: String,
}

impl BuildError {
    fn new(kind: BuildErrorKind, message: impl Into<String>) -> Self {
        BuildError {
            kind,
            message: message.into(),
        }
    }

    /// The category of the misuse.
    pub fn kind(&self) -> BuildErrorKind {
        self.kind
    }

    /// The description of the misuse, without the category.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for BuildError {}

/// The category of a `BuildError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BuildErrorKind {
    /// An instruction, block or annotation was added before any function was opened.
    NoOpenFunction,
    /// An instruction was added while no block is open.
    NoOpenBlock,
    /// A body was added to an external function.
    ExternFunctionBody,
    /// An instruction was added after its block already ends with a terminator.
    AfterTerminator,
    /// A function with the same name is already defined.
    DuplicateFunction,
    /// A block with the same label is already defined in the function.
    DuplicateBlock,
    /// A named type with the same name is already declared.
    DuplicateType,
    /// A global with the same name is already declared.
    DuplicateGlobal,
    /// The same annotation is applied to a function twice.
    DuplicateAnnotation,
    /// The annotation has no text form in Lamina IR.
    InvalidAnnotation,
    /// Two annotations applied to one function cannot be used together.
    ConflictingAnnotation,
    /// A global's linkage or alignment is malformed, or its string initializer does not fit
    /// its type.
    InvalidGlobal,
    /// A string literal would appear in an instruction, which only global initializers may hold.
    StringLiteralOperand,
    /// A type does not satisfy the rule of the instruction it is used in.
    InvalidType,
    /// A call's argument count does not fit its declared variadic split.
    InvalidArgument,
    /// A block does not end with a terminator, found by `build()` or `try_build()`.
    UnterminatedBlock,
    /// The entry block names a label the function never defines, found by `build()` or `try_build()`.
    UndefinedEntryBlock,
}

impl Default for IRBuilder<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> IRBuilder<'a> {
    /// Creates a new empty IR builder
    ///
    /// Initializes a builder with no functions or blocks. Misuse panics.
    pub fn new() -> Self {
        IRBuilder {
            module: Module::new(),
            functions: Vec::new(),
            current_function: None,
            current_block: None,
            temp_var_counter: 0,
            fallible: false,
            error: None,
        }
    }

    /// Creates a new empty IR builder that records misuse instead of panicking
    ///
    /// The first misuse is kept, and every later call does nothing. `try_build()` returns it.
    /// `build()` on such a builder panics with the recorded message.
    pub fn new_fallible() -> Self {
        IRBuilder {
            fallible: true,
            ..Self::new()
        }
    }

    /// Generates a unique temporary variable name
    ///
    /// Returns: A fresh variable name in the format "temp_N" where N is a counter
    ///
    /// Use this when you need a variable but don't care about its specific name.
    /// Each call returns a new name that won't conflict with previous ones.
    pub fn temp_var(&mut self) -> String {
        let var = format!("temp_{}", self.temp_var_counter);
        self.temp_var_counter += 1;
        var
    }

    /// Runs one step that may fail, and does nothing once a fallible builder has recorded an error.
    pub(super) fn step(
        &mut self,
        step: impl FnOnce(&mut Self) -> Result<(), BuildError>,
    ) -> &mut Self {
        if self.error.is_none()
            && let Err(error) = step(self)
        {
            self.fail(error);
        }
        self
    }

    /// Panics with the error, or records it when the builder is fallible and has no error yet.
    fn fail(&mut self, error: BuildError) {
        if !self.fallible {
            panic!("{error}");
        }
        self.error.get_or_insert(error);
    }

    /// Adds a raw instruction to the current block
    ///
    /// Panics when no block is open, when the current function is an extern declaration,
    /// or when the instruction would print a string literal, which only global initializers
    /// may contain.
    pub(super) fn inst(&mut self, instruction: Instruction<'a>) -> &mut Self {
        self.step(|builder| builder.push_instruction(instruction))
    }

    fn push_instruction(&mut self, instruction: Instruction<'a>) -> Result<(), BuildError> {
        let text = instruction.to_string();
        require(
            !text.contains('"'),
            BuildErrorKind::StringLiteralOperand,
            || format!("string literals only initialize globals, not instruction operands: {text}"),
        )?;
        let open_block = self.current_block;
        let function = self.open_body()?;
        let block = open_block.ok_or_else(|| {
            BuildError::new(
                BuildErrorKind::NoOpenBlock,
                format!("no open block in '@{}'; call block() first", function.name),
            )
        })?;
        let pending = &mut function.blocks[block];
        require(
            pending
                .instructions
                .last()
                .is_none_or(|last| !last.is_terminator()),
            BuildErrorKind::AfterTerminator,
            || {
                format!(
                    "block '{}' already ends with a terminator; start a new block with block()",
                    pending.name
                )
            },
        )?;
        pending.instructions.push(instruction);
        Ok(())
    }

    /// Returns the function currently being built, or fails when none is open.
    pub(super) fn open_function(&mut self) -> Result<&mut PendingFunction<'a>, BuildError> {
        let index = self.current_function.ok_or_else(|| {
            BuildError::new(
                BuildErrorKind::NoOpenFunction,
                "no open function; call function() or external_function() first",
            )
        })?;
        Ok(&mut self.functions[index])
    }

    /// Returns the function currently being built, or fails when none is open or it is extern.
    pub(super) fn open_body(&mut self) -> Result<&mut PendingFunction<'a>, BuildError> {
        let function = self.open_function()?;
        require(
            !function.is_extern,
            BuildErrorKind::ExternFunctionBody,
            || {
                format!(
                    "extern function '@{}' has no body; declare the body with function() instead",
                    function.name
                )
            },
        )?;
        Ok(function)
    }

    /// Finalizes and returns the complete IR module
    ///
    /// This method converts all the accumulated function and block data
    /// into a complete Module object that can be used for code generation,
    /// optimization, or serialization.
    ///
    /// Call this method only once all functions and instructions have been added.
    ///
    /// Returns: A Module object containing all defined functions and blocks
    ///
    /// Panics when a function's entry block was set to a label that was never defined, or
    /// when a block does not end with a terminator. On a fallible builder, panics with the
    /// first recorded misuse.
    pub fn build(&mut self) -> Module<'a> {
        self.try_build().unwrap_or_else(|error| panic!("{error}"))
    }

    /// Finalizes the module, or returns the first misuse instead of panicking
    ///
    /// Returns the misuse a fallible builder recorded, or the error `build()` would panic with.
    pub fn try_build(&mut self) -> Result<Module<'a>, BuildError> {
        if let Some(error) = &self.error {
            return Err(error.clone());
        }
        let mut module = self.module.clone();
        for function in &self.functions {
            for block in &function.blocks {
                let terminated = block
                    .instructions
                    .last()
                    .is_some_and(|last| last.is_terminator());
                require(terminated, BuildErrorKind::UnterminatedBlock, || {
                    format!(
                        "block '{}' of '@{}' must end with a terminator (ret, jmp, br, switch, unreachable)",
                        block.name, function.name
                    )
                })?;
            }
            let basic_blocks: HashMap<&'a str, BasicBlock<'a>> = function
                .blocks
                .iter()
                .map(|block| {
                    (
                        block.name,
                        BasicBlock {
                            instructions: block.instructions.clone(),
                            instruction_locs: Vec::new(),
                        },
                    )
                })
                .collect();
            let entry_block = if function.is_extern {
                ""
            } else {
                let entry = function.entry.unwrap_or("entry");
                require(
                    basic_blocks.contains_key(entry),
                    BuildErrorKind::UndefinedEntryBlock,
                    || {
                        format!(
                            "entry block '{entry}' of '@{}' is not defined",
                            function.name
                        )
                    },
                )?;
                entry
            };
            module.functions.insert(
                function.name,
                Function {
                    name: function.name,
                    signature: function.signature.clone(),
                    annotations: function.annotations.clone(),
                    basic_blocks,
                    entry_block,
                    decl_loc: None,
                },
            );
        }
        Ok(module)
    }
}

/// Succeeds when `condition` holds, and otherwise fails with `kind` and the message.
pub(super) fn require(
    condition: bool,
    kind: BuildErrorKind,
    message: impl FnOnce() -> String,
) -> Result<(), BuildError> {
    if condition {
        Ok(())
    } else {
        Err(BuildError::new(kind, message()))
    }
}
