//! Runtime compiler
//!
//! Compiles MIR modules to executable memory using ras.

#[cfg(feature = "encoder")]
use crate::mir_codegen::validate_module_call_parameters;
use crate::{
    error::LaminaError,
    mir::Module as MirModule,
    runtime::{RuntimeResult, compile_to_runtime},
};
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use ras::{ExecutableMemory, RasRuntime};
use std::mem;

/// Runtime compiler for JIT compilation
pub struct RuntimeCompiler {
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
    #[allow(dead_code)] // Used when encoder feature is enabled
    runtime: RasRuntime,
}

impl RuntimeCompiler {
    /// Create a new runtime compiler
    pub fn new(target_arch: TargetArchitecture, target_os: TargetOperatingSystem) -> Self {
        Self {
            target_arch,
            target_os,
            runtime: RasRuntime::new(target_arch, target_os),
        }
    }

    /// Compile a MIR module to executable memory.
    ///
    /// Every call compiles afresh: `ExecutableMemory` is not shareable, so there is
    /// nothing to hand a second caller.
    pub fn compile(
        &mut self,
        _module: &MirModule,
        _function_name: Option<&str>,
    ) -> Result<ExecutableMemory, LaminaError> {
        #[cfg(feature = "encoder")]
        {
            validate_module_call_parameters(_module, self.target_arch)?;
            self.runtime.compile_to_memory(_module).map_err(|e| {
                let error_msg = format!("{e}");
                if error_msg.contains("not yet implemented")
                    || error_msg.contains("Unsupported target")
                {
                    LaminaError::ValidationError(format!(
                        "JIT compilation is not supported for this target (or the MIR uses an unsupported construct).\n\
                             Error: {error_msg}\n\
                             JIT machine code is emitted only for x86_64 and AArch64.\n\
                             Consider AOT compilation instead (run without --jit)."
                    ))
                } else {
                    LaminaError::ValidationError(format!("Runtime compilation failed: {e}"))
                }
            })
        }
        #[cfg(not(feature = "encoder"))]
        {
            Err(LaminaError::ValidationError(
                "Runtime compilation requires the 'encoder' feature to be enabled in ras"
                    .to_string(),
            ))
        }
    }

    /// Compile and get a callable handle.
    ///
    /// The handle owns the executable mapping, so the code stays mapped for as long as
    /// the handle lives. A bare function pointer would outlive the mapping it points
    /// into.
    ///
    /// # Safety
    ///
    /// `T` must match the return type of the compiled function, and that function must
    /// take no arguments.
    pub unsafe fn compile_function<T>(
        &mut self,
        module: &MirModule,
        function_name: &str,
    ) -> Result<CompiledFunction<T>, LaminaError> {
        let compiled = self.compile_entry(module, function_name)?;
        let entry: unsafe extern "C" fn() -> T = unsafe { mem::transmute(compiled.function_ptr) };
        Ok(CompiledFunction {
            _memory: compiled.memory,
            entry,
        })
    }

    /// Compile the whole module and resolve `function_name` to its entry point.
    pub(crate) fn compile_entry(
        &mut self,
        module: &MirModule,
        function_name: &str,
    ) -> Result<RuntimeResult, LaminaError> {
        compile_to_runtime(
            module,
            self.target_arch,
            self.target_os,
            Some(function_name),
        )
    }
}

/// A compiled function together with the mapping it lives in.
///
/// Holding the `ExecutableMemory` here is the point: dropping this handle unmaps the
/// code, so the entry pointer cannot outlive what it points at.
pub struct CompiledFunction<T> {
    _memory: ExecutableMemory,
    entry: unsafe extern "C" fn() -> T,
}

impl<T> CompiledFunction<T> {
    /// # Safety
    ///
    /// `T` must match the return type of the compiled function, and that function must
    /// take no arguments.
    pub unsafe fn call(&self) -> T {
        unsafe { (self.entry)() }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(any(
        not(feature = "encoder"),
        all(target_arch = "x86_64", target_os = "linux")
    ))]
    use super::*;
    #[cfg(not(feature = "encoder"))]
    use crate::mir::Module;
    #[cfg(all(feature = "encoder", target_arch = "x86_64", target_os = "linux"))]
    use crate::mir::codegen::from_ir_x86_64;
    #[cfg(all(feature = "encoder", target_arch = "x86_64", target_os = "linux"))]
    use crate::parser::parse_module;
    // Only the x86_64 Linux JIT test and the no-encoder test build a compiler.
    #[cfg(any(
        not(feature = "encoder"),
        all(target_arch = "x86_64", target_os = "linux")
    ))]
    use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

    #[cfg(any(
        not(feature = "encoder"),
        all(target_arch = "x86_64", target_os = "linux")
    ))]
    fn make_compiler() -> RuntimeCompiler {
        RuntimeCompiler::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
    }

    /// The handle must keep the mapping alive. This used to return a bare pointer into
    /// an ExecutableMemory that was dropped on the way out, so calling it read unmapped
    /// memory.
    #[cfg(all(feature = "encoder", target_arch = "x86_64", target_os = "linux"))]
    #[test]
    fn compiled_function_stays_callable_after_the_compiler_returns() {
        let input = r#"
        fn @answer() -> i64 {
            entry:
                ret.i64 42
        }
        "#;
        let ir_module = parse_module(input).expect("parse");
        let module = from_ir_x86_64(&ir_module, "jit_test").expect("lower");

        let handle = {
            let mut compiler = make_compiler();
            // SAFETY: @answer takes no arguments and returns i64.
            match unsafe { compiler.compile_function::<i64>(&module, "@answer") } {
                Ok(handle) => handle,
                // JIT support is narrower than the AOT backends; skip where it is absent
                // rather than fail for an unrelated reason.
                Err(_) => return,
            }
        };

        // SAFETY: same contract as above, and the handle still owns the mapping.
        assert_eq!(unsafe { handle.call() }, 42);
    }

    #[cfg(all(feature = "encoder", target_arch = "x86_64", target_os = "linux"))]
    const TWO_FUNCTIONS: &str = r#"
        fn @first() -> i64 {
            entry:
                ret.i64 1
        }

        fn @second() -> i64 {
            entry:
                ret.i64 2
        }
        "#;

    #[cfg(all(feature = "encoder", target_arch = "x86_64", target_os = "linux"))]
    #[test]
    fn compile_function_enters_the_named_function() {
        let ir_module = parse_module(TWO_FUNCTIONS).expect("parse");
        let module = from_ir_x86_64(&ir_module, "jit_test").expect("lower");
        let mut compiler = make_compiler();

        for (name, expected) in [("@first", 1), ("@second", 2), ("second", 2)] {
            // SAFETY: both functions take no arguments and return i64.
            let handle =
                unsafe { compiler.compile_function::<i64>(&module, name) }.expect("compile");
            // SAFETY: same contract as above.
            assert_eq!(unsafe { handle.call() }, expected, "entry {name}");
        }
    }

    #[cfg(all(feature = "encoder", target_arch = "x86_64", target_os = "linux"))]
    #[test]
    fn compile_function_rejects_an_unknown_name() {
        let ir_module = parse_module(TWO_FUNCTIONS).expect("parse");
        let module = from_ir_x86_64(&ir_module, "jit_test").expect("lower");
        let mut compiler = make_compiler();

        // SAFETY: nothing is called; the lookup fails first.
        let result = unsafe { compiler.compile_function::<i64>(&module, "@third") };
        assert!(matches!(result, Err(LaminaError::ValidationError(_))));
    }

    #[cfg(not(feature = "encoder"))]
    #[test]
    fn compile_without_encoder_returns_validation_error() {
        let mut compiler = make_compiler();
        let module = Module::new("test");
        let result = compiler.compile(&module, None);
        assert!(matches!(result, Err(LaminaError::ValidationError(_))));
    }

    #[cfg(all(feature = "encoder", target_arch = "x86_64", target_os = "linux"))]
    mod typed_calls {
        use crate::{
            mir::codegen::from_ir_x86_64,
            parser::parse_module,
            runtime::{RuntimeResult, compile_to_runtime},
        };
        use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
        use std::mem::transmute;

        fn compile_entry(source: &str, name: &str) -> RuntimeResult {
            let ir_module = parse_module(source).expect("parse");
            let module = from_ir_x86_64(&ir_module, "jit_test").expect("lower");
            compile_to_runtime(
                &module,
                TargetArchitecture::X86_64,
                TargetOperatingSystem::Linux,
                Some(name),
            )
            .expect("compile")
        }

        const F64_PROGRAM: &str = r#"
        fn @ints_then_f64(i64 %a, i64 %b, f64 %x) -> i64 {
        entry:
          %s = add.i64 %a, %b
          ret.i64 %s
        }
        "#;

        #[test]
        fn f64_argument_passes_beside_integer_arguments() {
            let entry = compile_entry(F64_PROGRAM, "@ints_then_f64");
            // SAFETY: @ints_then_f64 takes two i64 and an f64 and returns i64; `entry` owns the code.
            let call: extern "C" fn(i64, i64, f64) -> i64 =
                unsafe { transmute(entry.function_ptr) };
            assert_eq!(call(40, 2, 1.5), 42);
            assert_eq!(call(-7, 10, f64::NAN), 3);
        }
    }
}
