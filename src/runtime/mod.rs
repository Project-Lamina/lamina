//! Runtime compilation and execution
//!
//! Runtime code generation (JIT compilation) using ras,
//! with optional sandboxing for secure execution.

pub mod c_abi_dynamic;
pub mod compiler;
pub mod executor;
#[cfg(feature = "encoder")]
pub mod macro_helpers;
pub mod sandbox;

pub use compiler::RuntimeCompiler;
pub use executor::execute_jit_function;
#[cfg(feature = "encoder")]
pub use macro_helpers::compile_lir_internal;
pub use sandbox::{Sandbox, SandboxConfig};

#[cfg(feature = "encoder")]
use std::env;
#[cfg(all(feature = "encoder", target_arch = "aarch64"))]
use std::{cmp::min, slice};

#[cfg(feature = "encoder")]
use crate::mir_codegen::validate_module_call_parameters;
use crate::{error::LaminaError, mir::Module as MirModule};
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

// Re-export ras types for convenience
#[cfg(feature = "encoder")]
use ras::assembler::RasAssembler;
pub use ras::{ExecutableMemory, RasRuntime};

/// Runtime compilation result
#[non_exhaustive]
pub struct RuntimeResult {
    /// Executable memory containing compiled code
    pub memory: ExecutableMemory,
    /// Function pointer (unsafe - caller must ensure signature matches)
    pub function_ptr: *const u8,
}

/// Compile MIR module to executable memory using runtime compilation
pub fn compile_to_runtime(
    _module: &MirModule,
    _target_arch: TargetArchitecture,
    _target_os: TargetOperatingSystem,
    _function_name: Option<&str>,
) -> Result<RuntimeResult, LaminaError> {
    #[cfg(feature = "encoder")]
    {
        validate_module_call_parameters(_module, _target_arch)?;

        // Keep JIT output quiet by default (release-friendly). Enable with `LAMINA_JIT_DEBUG=1`.
        let jit_debug = env::var_os("LAMINA_JIT_DEBUG").is_some();

        let mut assembler = RasAssembler::new(_target_arch, _target_os).map_err(|e| {
            LaminaError::ValidationError(format!("Failed to create assembler: {e}"))
        })?;

        // Always compile all functions (needed for internal function calls)
        let (code, function_offsets) = assembler
            .compile_mir_to_binary_function(_module, None)
            .map_err(|e| {
                LaminaError::ValidationError(format!("Runtime compilation failed: {e}"))
            })?;

        // Find the function offset for the requested function
        let function_offset = if let Some(name) = _function_name {
            let found = function_offsets
                .get::<str>(name)
                .or_else(|| {
                    if let Some(stripped) = name.strip_prefix('@') {
                        function_offsets.get(stripped)
                    } else {
                        function_offsets.get(&format!("@{name}"))
                    }
                })
                .copied();
            let Some(offset) = found else {
                let mut available: Vec<&str> =
                    function_offsets.keys().map(String::as_str).collect();
                available.sort_unstable();
                return Err(LaminaError::ValidationError(format!(
                    "JIT entry function '{name}' is not in the module (functions: {})",
                    available.join(", ")
                )));
            };
            if jit_debug {
                eprintln!(
                    "[JIT-DEBUG] Function '{}' offset: {:?}, available: {:?}",
                    name,
                    offset,
                    function_offsets.keys().collect::<Vec<_>>()
                );
            }
            Some(offset)
        } else {
            // If no function name specified, use the first function (offset 0)
            Some(0)
        };

        // Allocate writable memory
        let mut memory = ExecutableMemory::allocate_writable(code.len())
            .map_err(|e| LaminaError::ValidationError(format!("Memory allocation failed: {e}")))?;

        // Write code
        memory
            .write_code(&code)
            .map_err(|e| LaminaError::ValidationError(format!("Failed to write code: {e}")))?;

        // Make executable
        memory.make_executable().map_err(|e| {
            LaminaError::ValidationError(format!("Failed to make memory executable: {e}"))
        })?;

        // Get function pointer, adjusting for function offset if specified
        let base_ptr = memory.code_start();
        let function_ptr = if let Some(offset) = function_offset {
            // Ensure offset is 4-byte aligned for AArch64
            #[cfg(target_arch = "aarch64")]
            {
                if offset % 4 != 0 {
                    return Err(LaminaError::ValidationError(format!(
                        "Function offset {} is not 4-byte aligned (required for AArch64)",
                        offset
                    )));
                }
            }
            let adjusted = (base_ptr as usize + offset) as *const u8;
            if jit_debug {
                eprintln!(
                    "[JIT-DEBUG] Function pointer: base={base_ptr:p}, offset={offset}, adjusted={adjusted:p}"
                );
            }

            // Debug: Print first few instruction bytes at the function pointer
            #[cfg(target_arch = "aarch64")]
            {
                if jit_debug {
                    unsafe {
                        let remaining = code.len().saturating_sub(offset);
                        let bytes = slice::from_raw_parts(adjusted, min(128, remaining));
                        eprintln!(
                            "[JIT-DEBUG] First 32 bytes at function: {:02x?}",
                            bytes.iter().take(32).collect::<Vec<_>>()
                        );

                        // Check alignment
                        if !(adjusted as usize).is_multiple_of(4) {
                            eprintln!(
                                "[JIT-DEBUG][WARNING] Function pointer is not 4-byte aligned! Address: {:p}, alignment: {} bytes",
                                adjusted,
                                (adjusted as usize) % 4
                            );
                        } else {
                            eprintln!("[JIT-DEBUG] Function pointer is 4-byte aligned");
                        }

                        // Check last 4 bytes for RET instruction (should be c0 03 5f d6 for ret x30)
                        if bytes.len() >= 4 {
                            let last_4 = &bytes[bytes.len().saturating_sub(4)..];
                            eprintln!(
                                "[JIT-DEBUG] Last 4 bytes (RET instruction): {:02x?}",
                                last_4
                            );
                            let ret_inst =
                                u32::from_le_bytes([last_4[0], last_4[1], last_4[2], last_4[3]]);
                            eprintln!(
                                "[JIT-DEBUG] RET instruction value: 0x{:08x} (expected: 0xd65f03c0 for ret x30)",
                                ret_inst
                            );
                        }

                        // Decode first few instructions
                        eprintln!("[JIT-DEBUG] Decoding first 16 instructions (64 bytes):");
                        for i in 0..min(16, bytes.len() / 4) {
                            let inst_bytes = &bytes[i * 4..(i + 1) * 4];
                            let inst = u32::from_le_bytes([
                                inst_bytes[0],
                                inst_bytes[1],
                                inst_bytes[2],
                                inst_bytes[3],
                            ]);
                            // Decode instruction type
                            let opcode = (inst >> 26) & 0x3F;
                            let opcode_top = (inst >> 28) & 0xF;
                            let bits_29_27 = (inst >> 27) & 0x7;
                            let bits_25_24 = (inst >> 24) & 0x3;
                            let inst_type = if (inst >> 25) & 0x7F == 0b1101011 {
                                // RET/BR: [31:25]=1101011, [24]=0, [23:21]=010
                                "RET/BR"
                            } else if opcode == 0b100101 {
                                "BL"
                            } else if opcode_top == 0b00 && (inst >> 27) & 0x1 == 1 {
                                // STP/LDP: [31:30]=00, [27]=1
                                if (inst >> 28) & 0x1 == 0 {
                                    "STP"
                                } else {
                                    "LDP"
                                }
                            } else if opcode_top == 0b00 && (inst >> 27) & 0x1 == 0 {
                                // LDP (post-index): [31:30]=00, [28]=1, [27]=0
                                if (inst >> 28) & 0x1 == 1 {
                                    "LDP"
                                } else {
                                    "STP"
                                }
                            } else if bits_29_27 == 0b010
                                && (inst >> 23) & 0x3F == 0b100010
                                && opcode_top != 0b00
                            {
                                // ADD/SUB (immediate): [29:27]=010, [28:23]=100010, but not LDP/STP
                                if (inst >> 30) & 0x1 == 0 {
                                    "ADD(imm)"
                                } else {
                                    "SUB(imm)"
                                }
                            } else if bits_29_27 == 0b001
                                && (inst >> 23) & 0x3F == 0b010110
                                && (inst >> 30) & 0x3 == 0b10
                            {
                                // ADD (register, shifted register): [31:30]=10, [29:27]=001, [28:24]=01011, [23:22]=00
                                // This is ADD with no shift
                                "ADD(reg)"
                            } else if bits_29_27 == 0b010
                                && (inst >> 23) & 0x3F == 0b100010
                                && (inst >> 30) & 0x3 == 0b10
                            {
                                // ADD/SUB (register, extended): [31:30]=10, [29:27]=010, [28:23]=100010
                                if (inst >> 30) & 0x1 == 0 {
                                    "ADD(reg,ext)"
                                } else {
                                    "SUB(reg,ext)"
                                }
                            } else if bits_29_27 == 0b010 && (inst >> 23) & 0x3F == 0b100101 {
                                // MOVZ/MOVK: [29:27]=010, [28:23]=100101
                                if (inst >> 21) & 0x3 == 0b00 {
                                    "MOVZ"
                                } else {
                                    "MOVK"
                                }
                            } else if bits_29_27 == 0b111 && (inst >> 26) & 0x1 == 0 {
                                // STR/LDR unscaled immediate: [29:27]=111, [26]=0
                                if bits_25_24 == 0b00 {
                                    "STR(unscaled)"
                                } else if bits_25_24 == 0b01 {
                                    "LDR(unscaled)"
                                } else {
                                    "STR/LDR(scaled)"
                                }
                            } else if bits_29_27 == 0b001
                                && (inst >> 23) & 0x3F == 0b010100
                                && (inst >> 30) & 0x3 == 0b11
                            {
                                // EOR/XOR (register): [31:30]=11, [29:27]=001, [28:23]=010100
                                "EOR/XOR"
                            } else if (inst >> 31) & 0x1 == 1 && (inst >> 29) & 0x3 == 0b01 {
                                "MOV/ORR"
                            } else {
                                "UNKNOWN"
                            };
                            eprintln!("  [{}] 0x{:08x} ({})", i, inst, inst_type);
                        }
                    }
                }
            }

            adjusted
        } else {
            if jit_debug {
                eprintln!("[JIT-DEBUG] Function pointer: base={base_ptr:p}, no offset");
            }
            base_ptr
        };

        Ok(RuntimeResult {
            memory,
            function_ptr,
        })
    }
    #[cfg(not(feature = "encoder"))]
    {
        Err(LaminaError::ValidationError(
            "Runtime compilation requires the 'encoder' feature to be enabled".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "encoder"))]
    use super::*;
    #[cfg(not(feature = "encoder"))]
    use crate::mir::Module;
    #[cfg(not(feature = "encoder"))]
    use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

    #[cfg(not(feature = "encoder"))]
    #[test]
    fn compile_to_runtime_without_encoder_returns_error() {
        let module = Module::new("test");
        let result = compile_to_runtime(
            &module,
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Linux,
            None,
        );
        assert!(matches!(result, Err(LaminaError::ValidationError(_))));
    }

    /// Builtins resolved from the environment.
    mod builtins_env {
        use crate::test_support::TempDir;
        use crate::test_support::lamina_binary;
        use std::{fs, path::PathBuf, process, process::Command};

        const PROGRAM: &str = r#"
        @extern
        fn @malloc(i64 %n) -> ptr
        @extern
        fn @free(ptr %p) -> void
        @extern
        fn @memcpy(ptr %d, ptr %s, i64 %n) -> ptr
        @extern
        fn @memmove(ptr %d, ptr %s, i64 %n) -> ptr
        @extern
        fn @memset(ptr %d, i32 %v, i64 %n) -> ptr

        fn @main() -> i32 {
          entry:
            %p = call @malloc(8)
            %src = call @malloc(8)
            call @memcpy(%p, %src, 8)
            call @memmove(%p, %src, 8)
            call @memset(%p, 0, 8)
            %w = writebyte 55
            call @free(%p)
            ret.i32 0
        }
        "#;

        const OVERRIDES: &str =
            "malloc=k_alloc,dealloc=k_free,memcpy=k_cpy,memmove=k_mv,memset=k_set";

        fn run_compile(
            tag: &str,
            target: &str,
            builtins: Option<&str>,
        ) -> (process::Output, PathBuf, TempDir) {
            let guard = TempDir::new("lamina-builtins");
            let dir = guard.path();
            let stem = format!("lamina-builtins-{}-{tag}", process::id());
            let input = dir.join(format!("{stem}.lamina"));
            let output = dir.join(format!("{stem}.s"));
            fs::write(&input, PROGRAM).expect("write program");

            let mut command = Command::new(lamina_binary());
            command
                .arg(&input)
                .args(["--emit-asm", "--target", target, "-o"])
                .arg(&output)
                .env_remove("LAMINA_BUILTINS");
            if let Some(value) = builtins {
                command.env("LAMINA_BUILTINS", value);
            }
            let run = command.output().expect("run lamina");
            (run, output, guard)
        }

        fn compile_asm(tag: &str, target: &str, builtins: Option<&str>) -> String {
            let (run, output, _guard) = run_compile(tag, target, builtins);
            assert!(
                run.status.success(),
                "{}",
                String::from_utf8_lossy(&run.stderr)
            );

            fs::read_to_string(&output).expect("read asm")
        }

        fn refusal(tag: &str, builtins: &str) -> String {
            let (run, _output, _guard) = run_compile(tag, "x86_64_linux", Some(builtins));
            assert!(
                !run.status.success(),
                "an override the compiler cannot apply must not compile"
            );
            String::from_utf8_lossy(&run.stderr).into_owned()
        }

        #[test]
        fn default_output_uses_c_runtime_names() {
            let asm = compile_asm("default-linux", "x86_64_linux", None);
            for name in ["malloc", "memcpy", "memmove", "memset", "free"] {
                assert!(asm.contains(&format!("call {name}\n")), "{name}:\n{asm}");
            }
            assert!(!asm.contains("k_"), "{asm}");
        }

        #[test]
        fn lamina_builtins_retargets_every_runtime_call_x86_64() {
            let asm = compile_asm("override-linux", "x86_64_linux", Some(OVERRIDES));
            for name in ["k_alloc", "k_free", "k_cpy", "k_mv", "k_set"] {
                assert!(asm.contains(&format!("call {name}\n")), "{name}:\n{asm}");
            }
            for name in ["malloc", "memcpy", "memmove", "memset", "free"] {
                assert!(!asm.contains(&format!("call {name}\n")), "{name}:\n{asm}");
            }
        }

        #[test]
        fn lamina_builtins_retargets_every_runtime_call_aarch64_macos() {
            let asm = compile_asm("override-aarch64-macos", "aarch64_macos", Some(OVERRIDES));
            for name in ["_k_alloc", "_k_free", "_k_cpy", "_k_mv", "_k_set"] {
                assert!(asm.contains(&format!("{name}\n")), "{name}:\n{asm}");
            }
            assert!(!asm.contains("_malloc"), "{asm}");
        }

        #[test]
        fn an_unknown_builtin_name_is_refused_not_dropped() {
            let message = refusal("unknown-name", "malloc=k_alloc,mallocc=k_alloc");
            assert!(message.contains("mallocc"), "{message}");
            assert!(message.contains("malloc"), "{message}");
        }

        #[test]
        fn a_malformed_builtins_entry_is_refused() {
            let message = refusal("malformed", "malloc");
            assert!(message.contains("name=symbol"), "{message}");

            let empty = refusal("empty-side", "malloc=");
            assert!(empty.contains("empty"), "{empty}");
        }

        #[test]
        fn lamina_builtins_retargets_a_compiler_runtime_libcall() {
            const DIVIDE: &str = r#"
        fn @divide(i64 %a, i64 %b) -> i64 {
          entry:
            %q = div.i64 %a, %b
            ret.i64 %q
        }
        "#;
            let guard = TempDir::new("lamina-builtins-libcall");
            let dir = guard.path();
            let input = dir.join("libcall.lamina");
            let output = dir.join("libcall.s");
            fs::write(&input, DIVIDE).expect("write program");

            let run = Command::new(lamina_binary())
                .arg(&input)
                .args(["--emit-asm", "--target", "i686_linux", "-o"])
                .arg(&output)
                .env("LAMINA_BUILTINS", "__divdi3=k_divdi3")
                .output()
                .expect("run lamina");
            assert!(
                run.status.success(),
                "{}",
                String::from_utf8_lossy(&run.stderr)
            );
            let asm = fs::read_to_string(&output).expect("read asm");
            assert!(asm.contains("k_divdi3"), "{asm}");
            assert!(!asm.contains("__divdi3"), "{asm}");
        }
    }

    /// What the IO instructions promise.
    mod io_contract {
        #[cfg(any(
            all(target_os = "linux", target_arch = "x86_64"),
            all(target_os = "macos", target_arch = "aarch64")
        ))]
        mod native {
            use crate::test_support::TempDir;
            use crate::test_support::lamina_binary;
            use std::{
                fs,
                io::Write,
                path::{Path, PathBuf},
                process::{self, Command, Stdio},
            };

            enum Input {
                Empty,
                Closed,
                Pipe(&'static [u8]),
            }

            fn build(tag: &str, source: &str) -> (TempDir, PathBuf) {
                let guard = TempDir::new("lamina-io-contract");
                let stem = format!("lamina-io-contract-{}-{tag}", process::id());
                let input = guard.path().join(format!("{stem}.lamina"));
                let exe = guard.path().join(&stem);
                fs::write(&input, source).expect("write program");
                let run = Command::new(lamina_binary())
                    .arg(&input)
                    .arg("-o")
                    .arg(&exe)
                    .output()
                    .expect("run lamina");
                assert!(
                    run.status.success(),
                    "{}",
                    String::from_utf8_lossy(&run.stderr)
                );
                (guard, exe)
            }

            fn run(exe: &Path, input: Input, stdout_closed: bool) -> (i32, Vec<u8>) {
                let mut script = format!("exec '{}'", exe.display());
                match input {
                    Input::Empty => script.push_str(" </dev/null"),
                    Input::Closed => script.push_str(" <&-"),
                    Input::Pipe(_) => {}
                }
                if stdout_closed {
                    script.push_str(" >&-");
                }
                let mut child = Command::new("/bin/sh")
                    .arg("-c")
                    .arg(&script)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::null())
                    .spawn()
                    .expect("spawn program");
                if let Input::Pipe(bytes) = input {
                    child
                        .stdin
                        .take()
                        .expect("piped stdin")
                        .write_all(bytes)
                        .expect("write stdin");
                }
                let output = child.wait_with_output().expect("wait for program");
                (output.status.code().expect("normal exit"), output.stdout)
            }

            const WRITEBYTE: &str =
                "fn @main() -> i64 {\nentry:\n  %r = writebyte 65\n  ret.i64 %r\n}\n";

            #[test]
            fn writebyte_success_returns_one() {
                let (_dir, exe) = build("writebyte-ok", WRITEBYTE);
                assert_eq!(run(&exe, Input::Empty, false), (1, b"A".to_vec()));
            }

            #[test]
            fn writebyte_failure_returns_minus_two() {
                let (_dir, exe) = build("writebyte-fail", WRITEBYTE);
                assert_eq!(run(&exe, Input::Empty, true), (254, Vec::new()));
            }

            const WRITE: &str = "fn @main() -> i64 {\nentry:\n  %p = stackalloc.[4 x i8]\n  store.i8 %p, 79\n  %r = write %p, 3\n  ret.i64 %r\n}\n";

            #[test]
            fn write_success_returns_count() {
                let (_dir, exe) = build("write-ok", WRITE);
                let (code, stdout) = run(&exe, Input::Empty, false);
                assert_eq!(code, 3);
                assert_eq!(stdout.len(), 3);
            }

            #[test]
            fn write_zero_length_returns_zero() {
                let (_dir, exe) = build(
                    "write-zero",
                    "fn @main() -> i64 {\nentry:\n  %p = stackalloc.[4 x i8]\n  %r = write %p, 0\n  ret.i64 %r\n}\n",
                );
                assert_eq!(run(&exe, Input::Empty, false), (0, Vec::new()));
            }

            #[test]
            fn write_failure_returns_minus_two() {
                let (_dir, exe) = build("write-fail", WRITE);
                assert_eq!(run(&exe, Input::Empty, true), (254, Vec::new()));
            }

            const READ: &str = "fn @main() -> i64 {\nentry:\n  %p = stackalloc.[4 x i8]\n  %r = read %p, 1\n  ret.i64 %r\n}\n";

            #[test]
            fn read_success_returns_count() {
                let (_dir, exe) = build("read-ok", READ);
                assert_eq!(run(&exe, Input::Pipe(b"x"), false), (1, Vec::new()));
            }

            #[test]
            fn read_zero_length_returns_zero_without_touching_input() {
                let (_dir, exe) = build(
                    "read-zero",
                    "fn @main() -> i64 {\nentry:\n  %p = stackalloc.[4 x i8]\n  %r = read %p, 0\n  ret.i64 %r\n}\n",
                );
                assert_eq!(run(&exe, Input::Closed, false), (0, Vec::new()));
            }

            #[test]
            fn read_end_of_input_returns_minus_one() {
                let (_dir, exe) = build("read-eof", READ);
                assert_eq!(run(&exe, Input::Empty, false), (255, Vec::new()));
            }

            #[test]
            fn read_failure_returns_minus_two() {
                let (_dir, exe) = build("read-fail", READ);
                assert_eq!(run(&exe, Input::Closed, false), (254, Vec::new()));
            }

            const READBYTE: &str =
                "fn @main() -> i64 {\nentry:\n  %r = readbyte\n  ret.i64 %r\n}\n";

            #[test]
            fn readbyte_returns_the_byte() {
                let (_dir, exe) = build("readbyte-byte", READBYTE);
                assert_eq!(run(&exe, Input::Pipe(b"A"), false), (65, Vec::new()));
            }

            #[test]
            fn readbyte_nul_is_not_end_of_input() {
                let (_dir, exe) = build("readbyte-nul", READBYTE);
                assert_eq!(run(&exe, Input::Pipe(b"\0"), false), (0, Vec::new()));
            }

            #[test]
            fn readbyte_end_of_input_returns_minus_one() {
                let (_dir, exe) = build("readbyte-eof", READBYTE);
                assert_eq!(run(&exe, Input::Empty, false), (255, Vec::new()));
            }

            #[test]
            fn readbyte_failure_returns_minus_two() {
                let (_dir, exe) = build("readbyte-fail", READBYTE);
                assert_eq!(run(&exe, Input::Closed, false), (254, Vec::new()));
            }
        }
    }

    /// The JIT answers what the compiled program answers.
    mod jit_matches_aot {
        use crate::test_support::TempDir;
        use crate::test_support::lamina_binary;
        use std::{fs, process, process::Command};

        const HELPERS: &str = r#"
        fn @put_u64(u64 %n) -> void {
        entry:
          %q = div.u64 %n, 10
          %r = rem.u64 %n, 10
          %d = add.u64 %r, 48
          %z = eq.u64 %q, 0
          br %z, last, more
        more:
          call @put_u64(%q)
          jmp last
        last:
          %w = writebyte %d
          ret.void
        }

        fn @print_i64(i64 %n) -> void {
        entry:
          %neg = lt.i64 %n, 0
          %m = sub.i64 0, %n
          %s = select.i64 %neg, %m, %n
          %u = bitcast.i64.u64 %s
          br %neg, minus, digits
        minus:
          %c = writebyte 45
          jmp digits
        digits:
          call @put_u64(%u)
          %nl = writebyte 10
          ret.void
        }

        fn @show(i128 %v) -> i64 {
        entry:
          %lo = trunc.i128.i64 %v
          call @print_i64(%lo)
          %hs = shr.i128 %v, 64
          %hi = trunc.i128.i64 %hs
          call @print_i64(%hi)
          ret.i64 0
        }
        "#;

        const WIDE_DIVISION: &str = r#"
        fn @main() -> i32 {
        entry:
          %big = shl.i128 1, 100
          %d = add.i128 0, 1000000007
          %q = div.i128 %big, %d
          call @show(%q)
          %r = rem.i128 %big, %d
          call @show(%r)
          %ub0 = shl.i128 1, 127
          %ub = bitcast.i128.u128 %ub0
          %uq = div.u128 %ub, 3
          %uqs = bitcast.u128.i128 %uq
          call @show(%uqs)
          %ur = rem.u128 %ub, 3
          %urs = bitcast.u128.i128 %ur
          call @show(%urs)
          %nb = sub.i128 0, %big
          %sq = div.i128 %nb, %d
          call @show(%sq)
          %sr = rem.i128 %nb, %d
          call @show(%sr)
          ret.i32 0
        }
        "#;

        /// The corners where lamina's own routines and the host compiler runtime are known to
        /// part company: lamina saturates and returns zero for NaN, libgcc leaves them undefined.
        const FLOAT_CORNERS: &str = r#"
        fn @main() -> i32 {
        entry:
          %inf_b = add.u64 0, 9218868437227405312
          %inf = bitcast.u64.f64 %inf_b
          %nan_b = add.u64 0, 9221120237041090560
          %nan = bitcast.u64.f64 %nan_b
          %a = fptoint.f64.i128 %nan
          call @show(%a)
          %b = fptoint.f64.i128 %inf
          call @show(%b)
          %c = fptoint.f64.i128 1e40
          call @show(%c)
          %d = fptoint.f64.u128 -5.0
          %ds = bitcast.u128.i128 %d
          call @show(%ds)
          %e = fptoint.f64.u128 %nan
          %es = bitcast.u128.i128 %e
          call @show(%es)
          ret.i32 0
        }
        "#;

        fn outputs(tag: &str, program: &str) -> (String, String) {
            let guard = TempDir::new("lamina-jit-agreement");
            let dir = guard.path();
            let stem = format!("lamina-jit-{}-{tag}", process::id());
            let input = dir.join(format!("{stem}.lamina"));
            let exe = dir.join(&stem);
            fs::write(&input, format!("{program}{HELPERS}")).expect("write program");

            let build = Command::new(lamina_binary())
                .arg(&input)
                .arg("-o")
                .arg(&exe)
                .output()
                .expect("run lamina");
            assert!(
                build.status.success(),
                "{}",
                String::from_utf8_lossy(&build.stderr)
            );
            let aot = Command::new(&exe).output().expect("run compiled program");
            assert!(aot.status.success(), "compiled program failed: {aot:?}");

            let jit = Command::new(lamina_binary())
                .arg(&input)
                .arg("--jit")
                .output()
                .expect("run lamina --jit");
            assert!(
                jit.status.success(),
                "{}",
                String::from_utf8_lossy(&jit.stderr)
            );

            (
                String::from_utf8_lossy(&aot.stdout).into_owned(),
                String::from_utf8_lossy(&jit.stdout).into_owned(),
            )
        }

        #[cfg(any(
            all(target_os = "linux", target_arch = "x86_64"),
            all(target_os = "macos", target_arch = "aarch64")
        ))]
        #[test]
        fn jit_and_aot_agree_on_wide_division() {
            let (aot, jit) = outputs("div", WIDE_DIVISION);
            assert_eq!(aot, jit);
            assert!(!aot.trim().is_empty());
        }

        #[cfg(any(
            all(target_os = "linux", target_arch = "x86_64"),
            all(target_os = "macos", target_arch = "aarch64")
        ))]
        #[test]
        fn jit_and_aot_agree_on_float_to_wide_corners() {
            let (aot, jit) = outputs("corners", FLOAT_CORNERS);
            assert_eq!(aot, jit);
            let lines: Vec<&str> = aot.lines().collect();
            assert_eq!(
                lines,
                [
                    "0",
                    "0",
                    "-1",
                    "9223372036854775807",
                    "-1",
                    "9223372036854775807",
                    "0",
                    "0",
                    "0",
                    "0"
                ]
            );
        }

        #[cfg(any(
            all(target_os = "linux", target_arch = "x86_64"),
            all(target_os = "macos", target_arch = "aarch64")
        ))]
        #[test]
        fn jit_and_aot_agree_when_main_returns_i32() {
            let (aot, jit) = outputs(
                "narrow_main",
                "fn @main() -> i32 {\nentry:\n  call @print_i64(7)\n  ret.i32 0\n}\n",
            );
            assert_eq!(aot, jit);
            assert_eq!(aot, "7\n");
        }
    }
}
