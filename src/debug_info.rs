//! What a debugger is told about a module, built from MIR and handed to the assembler.
//!
//! The DWARF bytes are written by `ras`; this names the subprograms and the types in
//! terms the writer can encode. Nothing here states where a value lives at run time.
//! That needs the register allocator's result, which does not outlive the backends, and
//! a location guessed from anything less makes a debugger print a confident wrong
//! answer. A parameter therefore arrives with a name and a type and no location, which
//! a debugger reports as optimized out.
//!
//! When the backends do publish their allocation, a `location: Option<Vec<u8>>` on
//! [`ras::FormalParameter`] and a `locals` list beside `params` are where it attaches;
//! every subprogram already carries `DW_AT_frame_base`, so a frame-relative expression
//! has something to be relative to.

use lamina_mir::{
    Module as MirModule, Parameter, ScalarType, Signedness, SymbolVisibility, types::MirType,
};
use lamina_platform::TargetOperatingSystem;
use ras::{
    BaseType, DebugInfoRequest, DebugType, DeclSite, FormalParameter, ObjectWriteOptions,
    Subprogram,
};

/// Whether the object this target writes carries DWARF sections.
pub fn emits_dwarf(target_os: TargetOperatingSystem) -> bool {
    matches!(
        target_os,
        TargetOperatingSystem::Linux
            | TargetOperatingSystem::FreeBSD
            | TargetOperatingSystem::OpenBSD
            | TargetOperatingSystem::NetBSD
            | TargetOperatingSystem::DragonFly
            | TargetOperatingSystem::Redox
            | TargetOperatingSystem::MacOS
    )
}

/// Whether the object this target writes carries CodeView debug sections, which the
/// linker gathers into a PDB.
pub fn emits_codeview(target_os: TargetOperatingSystem) -> bool {
    target_os == TargetOperatingSystem::Windows
}

/// Mach-O labels a function's code with its name under a leading underscore, which the
/// debugger must not show and the assembler must be asked for by.
fn object_symbol(target_os: TargetOperatingSystem, name: &str) -> String {
    match target_os {
        TargetOperatingSystem::MacOS => format!("_{name}"),
        _ => name.to_string(),
    }
}

/// Fill in what the assembler needs to describe this module's functions. On a DWARF
/// target the symbol types and sizes go in whether or not debug info was asked for,
/// and the DWARF only under `-g`. COFF takes a symbol's type from its section and has
/// no symbol size, so there the whole request is `-g` only and a build without it is
/// byte-identical.
pub fn apply(
    options: &mut ObjectWriteOptions,
    target_os: TargetOperatingSystem,
    debug: bool,
    module: &MirModule,
) {
    if !emits_dwarf(target_os) && !(debug && emits_codeview(target_os)) {
        return;
    }
    options.function_symbols = function_symbols(module)
        .iter()
        .map(|name| object_symbol(target_os, name))
        .collect();
    if debug {
        options.debug_info = Some(describe(target_os, module));
    }
}

/// Names of the functions this module defines, for the assembler to mark `STT_FUNC` and
/// size. A declaration has no body here and is left out.
pub fn function_symbols(module: &MirModule) -> Vec<String> {
    let mut names: Vec<String> = module
        .functions
        .iter()
        .filter(|(name, func)| !module.is_external(name) && !func.blocks.is_empty())
        .map(|(name, _)| name.clone())
        .collect();
    names.sort_unstable();
    names
}

pub fn describe(target_os: TargetOperatingSystem, module: &MirModule) -> DebugInfoRequest {
    let mut subprograms: Vec<Subprogram> = module
        .functions
        .iter()
        .filter(|(name, func)| !module.is_external(name) && !func.blocks.is_empty())
        .map(|(name, func)| Subprogram {
            name: name.clone(),
            symbol: Some(object_symbol(target_os, name)).filter(|symbol| symbol != name),
            external: is_linkable(module, name, func.sig.visibility),
            decl: func.sig.decl.map(|loc| DeclSite {
                file: loc.file.get(),
                line: loc.line.get(),
            }),
            ret: func
                .sig
                .ret_aggregate
                .is_none()
                .then(|| {
                    func.sig
                        .ret_ty
                        .and_then(|ty| debug_type(ty, func.sig.ret_signedness))
                })
                .flatten(),
            params: func.sig.params.iter().map(parameter).collect(),
            variadic: func.sig.variadic,
        })
        .collect();
    subprograms.sort_by(|left, right| left.name.cmp(&right.name));
    DebugInfoRequest {
        producer: Some(format!("lamina {}", env!("CARGO_PKG_VERSION"))),
        subprograms,
    }
}

fn is_linkable(module: &MirModule, name: &str, visibility: Option<SymbolVisibility>) -> bool {
    name == "main"
        || module.exported_functions.contains(name)
        || matches!(
            visibility,
            Some(SymbolVisibility::Hidden | SymbolVisibility::Protected)
        )
}

/// An aggregate parameter keeps its name and loses its type: `reg` holds the address of
/// the callee's copy, and MIR records the aggregate's size and alignment but not its
/// members, so neither a struct nor the pointer would describe what the source declared.
fn parameter(param: &Parameter) -> FormalParameter {
    FormalParameter {
        name: param.name.clone(),
        ty: match param.aggregate {
            Some(_) => None,
            None => debug_type(param.ty, param.signedness),
        },
    }
}

fn debug_type(ty: MirType, signedness: Signedness) -> Option<DebugType> {
    let unsigned = signedness == Signedness::Unsigned;
    let scalar = match ty {
        MirType::Scalar(scalar) => scalar,
        MirType::Vector(_) => return None,
    };
    let base = match scalar {
        ScalarType::I1 => BaseType::Bool,
        ScalarType::I8 if unsigned => BaseType::U8,
        ScalarType::I8 => BaseType::I8,
        ScalarType::I16 if unsigned => BaseType::U16,
        ScalarType::I16 => BaseType::I16,
        ScalarType::I32 if unsigned => BaseType::U32,
        ScalarType::I32 => BaseType::I32,
        ScalarType::I64 if unsigned => BaseType::U64,
        ScalarType::I64 => BaseType::I64,
        ScalarType::F32 => BaseType::F32,
        ScalarType::F64 => BaseType::F64,
        ScalarType::Ptr | ScalarType::PtrIn(_) => return Some(DebugType::Pointer),
    };
    Some(DebugType::Base(base))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::codegen::from_ir_for_target;
    use crate::parser::parse_module;
    use lamina_platform::Target;
    use std::str::FromStr;

    const SOURCE: &str = r#"@extern fn @outside(i64 %n) -> i64

type !pair = struct { a: i64, b: i64 }

fn @scale(f64 %x, u32 %times) -> f64 {
  entry:
    ret.f64 %x
}

fn @widen(!pair %p) -> i64 {
  entry:
    ret.i64 0
}

fn @main() -> i64 {
  entry:
    ret.i64 0
}
"#;

    fn described() -> DebugInfoRequest {
        let target = Target::from_str("x86_64_linux").expect("target");
        let module = parse_module(SOURCE).expect("parse");
        let mir = from_ir_for_target(&module, "module", target.data_layout()).expect("lower");
        describe(TargetOperatingSystem::Linux, &mir)
    }

    fn subprogram(request: &DebugInfoRequest, name: &str) -> Subprogram {
        request
            .subprograms
            .iter()
            .find(|sub| sub.name == name)
            .unwrap_or_else(|| panic!("no subprogram for {name}"))
            .clone()
    }

    #[test]
    fn a_declaration_without_a_body_is_not_described() {
        let request = described();
        assert!(request.subprograms.iter().all(|sub| sub.name != "outside"));
        assert_eq!(request.subprograms.len(), 3);
    }

    #[test]
    fn a_function_carries_the_line_its_fn_keyword_is_on() {
        let request = described();
        let decl = subprogram(&request, "scale").decl.expect("decl");
        assert_eq!(decl.line, 5);
        assert_eq!(subprogram(&request, "main").decl.expect("decl").line, 15);
    }

    #[test]
    fn only_a_linkable_function_is_external() {
        let request = described();
        assert!(subprogram(&request, "main").external);
        assert!(!subprogram(&request, "scale").external);
    }

    #[test]
    fn a_parameter_keeps_its_source_name_and_signedness() {
        let params = subprogram(&described(), "scale").params;
        let names: Vec<Option<&str>> = params.iter().map(|p| p.name.as_deref()).collect();
        assert_eq!(names, vec![Some("x"), Some("times")]);
        assert_eq!(params[0].ty, Some(DebugType::Base(BaseType::F64)));
        assert_eq!(params[1].ty, Some(DebugType::Base(BaseType::U32)));
    }

    /// MIR records an aggregate's size and alignment but not its members, so neither
    /// a struct nor the pointer that carries it describes what the source declared.
    #[test]
    fn an_aggregate_parameter_is_named_and_left_untyped() {
        let params = subprogram(&described(), "widen").params;
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].name.as_deref(), Some("p"));
        assert_eq!(params[0].ty, None);
    }

    #[test]
    fn the_function_symbols_are_the_functions_with_a_body() {
        let target = Target::from_str("x86_64_linux").expect("target");
        let module = parse_module(SOURCE).expect("parse");
        let mir = from_ir_for_target(&module, "module", target.data_layout()).expect("lower");
        assert_eq!(function_symbols(&mir), vec!["main", "scale", "widen"]);
    }

    #[test]
    fn a_target_whose_object_carries_no_debug_info_is_described_to_no_one() {
        let target = Target::from_str("x86_64_linux").expect("target");
        let module = parse_module(SOURCE).expect("parse");
        let mir = from_ir_for_target(&module, "module", target.data_layout()).expect("lower");
        let mut options = ObjectWriteOptions::default();
        apply(&mut options, TargetOperatingSystem::Orbis, true, &mir);
        assert!(options.function_symbols.is_empty());
        assert!(options.debug_info.is_none());
    }

    /// COFF takes a symbol's type from its section and has no symbol size, so there is
    /// nothing for the assembler to fill in unless CodeView was asked for.
    #[test]
    fn a_windows_build_is_described_only_under_g() {
        let target = Target::from_str("x86_64_windows").expect("target");
        let module = parse_module(SOURCE).expect("parse");
        let mir = from_ir_for_target(&module, "module", target.data_layout()).expect("lower");

        let mut plain = ObjectWriteOptions::default();
        apply(&mut plain, TargetOperatingSystem::Windows, false, &mir);
        assert!(plain.function_symbols.is_empty());
        assert!(plain.debug_info.is_none());

        let mut debug = ObjectWriteOptions::default();
        apply(&mut debug, TargetOperatingSystem::Windows, true, &mir);
        assert_eq!(debug.function_symbols, vec!["main", "scale", "widen"]);
        let request = debug.debug_info.expect("a request");
        // COFF spells a function's symbol exactly as the source named it.
        assert!(request.subprograms.iter().all(|sub| sub.symbol.is_none()));
    }

    /// Mach-O labels a function's code with a leading underscore and a debugger must
    /// not show it, so the assembler is asked for one name and tells the debugger the
    /// other.
    #[test]
    fn a_mach_o_subprogram_keeps_its_source_name_and_names_its_underscored_symbol() {
        let target = Target::from_str("x86_64_macos").expect("target");
        let module = parse_module(SOURCE).expect("parse");
        let mir = from_ir_for_target(&module, "module", target.data_layout()).expect("lower");
        let mut options = ObjectWriteOptions::default();
        apply(&mut options, TargetOperatingSystem::MacOS, true, &mir);

        assert_eq!(options.function_symbols, vec!["_main", "_scale", "_widen"]);
        let request = options.debug_info.expect("a described module");
        let main = subprogram(&request, "main");
        assert_eq!(main.symbol.as_deref(), Some("_main"));
        assert!(
            request
                .subprograms
                .iter()
                .all(|sub| !sub.name.starts_with('_'))
        );
    }

    #[test]
    fn an_elf_subprogram_names_no_separate_symbol() {
        let request = described();
        assert!(request.subprograms.iter().all(|sub| sub.symbol.is_none()));
    }

    /// What a Windows debugger is told about a `-g` build.
    ///
    /// No PDB is written by the compiler: `.debug$S` and `.debug$T` go into the COFF
    /// object and the linker gathers them, which is what `link.exe` and `lld-link` are
    /// for. The end-to-end assertion here is therefore that `lld-link` builds a PDB from
    /// our objects and that the PDB answers the questions a debugger asks of it.
    mod codeview {
        use crate::api::{ObjectBuild, Target, compile_module_to_object_with};
        use crate::parser::parse_module;
        use crate::test_support::TempDir;
        use std::path::Path;
        use std::process::Command;
        use std::str::FromStr;
        use std::{env, fs};

        const SOURCE: &str = r#"fn @triple(i64 %n) -> i64 {
          entry:
            %r = mul.i64 %n, 3
            ret.i64 %r
        }

        fn @main() -> i64 {
          entry:
            %v = call @triple(7)
            ret.i64 0
        }
        "#;

        fn object(debug: bool) -> Vec<u8> {
            let module = parse_module(SOURCE).expect("parse");
            let target = Target::from_str("x86_64_windows").expect("target");
            let build = ObjectBuild {
                debug_lines: debug,
                ..ObjectBuild::default()
            };
            compile_module_to_object_with(&module, target, &build).expect("compile")
        }

        fn u16_at(bytes: &[u8], at: usize) -> u16 {
            u16::from_le_bytes(bytes[at..at + 2].try_into().expect("two bytes"))
        }

        fn u32_at(bytes: &[u8], at: usize) -> u32 {
            u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
        }

        fn section(object: &[u8], name: &str) -> Option<Vec<u8>> {
            let count = usize::from(u16_at(object, 2));
            let at = (0..count).map(|i| 20 + i * 40).find(|&at| {
                let raw = &object[at..at + 8];
                let end = raw.iter().position(|&b| b == 0).unwrap_or(8);
                raw[..end] == *name.as_bytes()
            })?;
            let size = u32_at(object, at + 16) as usize;
            let start = u32_at(object, at + 20) as usize;
            Some(object[start..start + size].to_vec())
        }

        #[test]
        fn a_build_without_g_carries_no_codeview() {
            let object = object(false);
            assert!(section(&object, ".debug$S").is_none());
            assert!(section(&object, ".debug$T").is_none());
        }

        #[test]
        fn a_g_build_carries_both_codeview_sections() {
            let object = object(true);
            for name in [".debug$S", ".debug$T"] {
                let section = section(&object, name).unwrap_or_else(|| panic!("no {name}"));
                assert_eq!(u32_at(&section, 0), 4, "{name} is not CV_SIGNATURE_C13");
            }
        }

        /// Mirrors the DWARF side: nothing outside the backends knows where a value lives
        /// after register allocation, so no `S_LOCAL` and no `S_DEFRANGE_*` may appear. A
        /// future guess trips this before it reaches a debugger.
        #[test]
        fn no_variable_location_is_stated() {
            let Some(pdbutil) = tool(&["llvm-pdbutil", "llvm-pdbutil-18", "llvm-pdbutil-19"])
            else {
                return;
            };
            let dir = TempDir::new("codeview-locations");
            let path = dir.path().join("program.obj");
            fs::write(&path, object(true)).expect("write");
            let dump = dump(&pdbutil, &["dump", "--symbols"], &path);
            assert!(dump.contains("S_GPROC32_ID"), "{dump}");
            assert!(!dump.contains("S_LOCAL"), "{dump}");
            assert!(!dump.contains("S_DEFRANGE"), "{dump}");
        }

        /// The whole point of emitting sections instead of writing a PDB: a linker has to be
        /// able to make one out of them. If this stops holding, leaving PDB generation to the
        /// linker is no longer a working plan.
        #[test]
        fn lld_link_builds_a_pdb_that_names_our_functions() {
            let Some(pdbutil) = tool(&["llvm-pdbutil", "llvm-pdbutil-18", "llvm-pdbutil-19"])
            else {
                return;
            };
            let Some(lld) = tool(&["lld-link", "lld-link-18", "lld-link-19"]) else {
                return;
            };
            let dir = TempDir::new("codeview-pdb");
            let object_path = dir.path().join("program.obj");
            let exe = dir.path().join("program.exe");
            let pdb = dir.path().join("program.pdb");
            fs::write(&object_path, object(true)).expect("write");

            let link = Command::new(lld)
                .args([
                    "/nologo",
                    "/entry:main",
                    "/subsystem:console",
                    "/nodefaultlib",
                    "/debug",
                ])
                .arg(format!("/pdb:{}", pdb.display()))
                .arg(format!("/out:{}", exe.display()))
                .arg(&object_path)
                .output()
                .expect("lld-link");
            assert!(
                link.status.success(),
                "{}",
                String::from_utf8_lossy(&link.stderr)
            );
            assert!(pdb.exists(), "lld-link produced no PDB");

            let dump = dump(
                &pdbutil,
                &["dump", "--symbols", "--types", "--ids", "-l"],
                &pdb,
            );
            // A procedure that reached the PDB has an address in a section of the image,
            // which only happens if the object's relocations were understood.
            assert!(dump.contains("S_GPROC32 [size = 44] `main`"), "{dump}");
            assert!(dump.contains("`triple`"), "{dump}");
            assert!(dump.contains("LF_FUNC_ID"), "{dump}");
            assert!(!dump.contains("addr = 0000:0000"), "{dump}");
            // The line table survived, mapping image offsets to the lines `triple`'s body is
            // on, which is what a breakpoint is placed by.
            assert!(dump.contains("line/addr entries = 2"), "{dump}");
            assert!(dump.contains("     3 "), "{dump}");
            assert!(!dump.contains("S_DEFRANGE"), "{dump}");
        }

        fn dump(tool: &str, args: &[&str], path: &Path) -> String {
            let output = Command::new(tool)
                .args(args)
                .arg(path)
                .output()
                .expect("dump");
            let text = String::from_utf8_lossy(&output.stdout).into_owned();
            assert!(
                output.status.success(),
                "{text}{}",
                String::from_utf8_lossy(&output.stderr)
            );
            text
        }

        /// Skips when the tool is missing, unless the suite was told to require it.
        fn tool(names: &[&str]) -> Option<String> {
            let found = names.iter().find(|name| {
                Command::new(name)
                    .arg("--version")
                    .output()
                    .is_ok_and(|output| output.status.success())
            });
            if let Some(name) = found {
                return Some((*name).to_string());
            }
            let required = env::var_os("LAMINA_REQUIRE_TOOLS").is_some_and(|value| value == "1");
            assert!(
                !required,
                "none of {names:?} is installed (LAMINA_REQUIRE_TOOLS=1)"
            );
            None
        }
    }

    /// A `-g` Mach-O object, read back with the LLVM tools.
    ///
    /// There is no Mac here. Every assertion is structural, and the strongest of them is
    /// that `dsymutil` itself links this object's DWARF when it is handed a debug map.
    /// Nothing below claims a debugger stopped at a breakpoint on Darwin.
    mod macho {
        use crate::api::{ObjectBuild, Target, compile_module_to_object_with};
        use crate::parser::parse_module;
        use crate::test_support::TempDir;
        use std::env;
        use std::fs;
        use std::process::Command;
        use std::str::FromStr;

        const SOURCE: &str = r#"fn @triple(i64 %n) -> i64 {
          entry:
            %r = mul.i64 %n, 3
            ret.i64 %r
        }

        fn @scale(f64 %x, u32 %k) -> f64 {
          entry:
            ret.f64 %x
        }

        fn @main() -> i64 {
          entry:
            %v = call @triple(7)
            ret.i64 %v
        }
        "#;

        /// Skips when the tool is missing, unless the suite was told to require it.
        fn tool(tool: &str) -> Option<String> {
            for name in [tool.to_string(), format!("{tool}-18"), format!("{tool}-19")] {
                let found = Command::new(&name)
                    .arg("--version")
                    .output()
                    .is_ok_and(|output| output.status.success());
                if found {
                    return Some(name);
                }
            }
            let required = env::var_os("LAMINA_REQUIRE_TOOLS").is_some_and(|value| value == "1");
            assert!(
                !required,
                "{tool} is not installed (LAMINA_REQUIRE_TOOLS=1)"
            );
            None
        }

        fn object(target: &str, debug: bool) -> Vec<u8> {
            let target = Target::from_str(target).expect("target string parses");
            let module = parse_module(SOURCE).expect("module parses");
            let build = ObjectBuild {
                debug_lines: debug,
                ..ObjectBuild::default()
            };
            compile_module_to_object_with(&module, target, &build).expect("object bytes")
        }

        fn dwarfdump(target: &str, switch: &str) -> Option<String> {
            let dwarfdump = tool("llvm-dwarfdump")?;
            let dir = TempDir::new("macho-debug-info");
            let path = dir.path().join("out.o");
            fs::write(&path, object(target, true)).expect("write object");
            let output = Command::new(dwarfdump)
                .arg(switch)
                .arg(&path)
                .output()
                .expect("llvm-dwarfdump");
            assert!(output.status.success(), "{switch} on {target}");
            Some(String::from_utf8_lossy(&output.stdout).into_owned())
        }

        #[test]
        fn both_mach_o_targets_pass_the_dwarf_verifier() {
            for target in ["x86_64_macos", "aarch64_macos"] {
                let Some(text) = dwarfdump(target, "--verify") else {
                    return;
                };
                assert!(text.contains("No errors."), "{target}: {text}");
            }
        }

        #[test]
        fn every_function_with_a_body_is_a_subprogram_at_its_own_address() {
            for target in ["x86_64_macos", "aarch64_macos"] {
                let Some(dump) = dwarfdump(target, "--debug-info") else {
                    return;
                };
                assert_eq!(dump.matches("DW_TAG_subprogram").count(), 3, "{dump}");
                for (name, line) in [("triple", 1), ("scale", 7), ("main", 12)] {
                    let die = dump
                        .split("DW_TAG_subprogram")
                        .find(|part| part.contains(&format!("DW_AT_name\t(\"{name}\")")))
                        .unwrap_or_else(|| panic!("no subprogram for {name} in {dump}"));
                    assert!(die.contains(&format!("DW_AT_decl_line\t({line})")), "{die}");
                    assert!(die.contains("DW_AT_frame_base"), "{die}");
                    assert!(die.contains("DW_AT_low_pc"), "{die}");
                }
                // The underscore belongs to the object label, not to what a debugger shows.
                assert!(!dump.contains("\"_main\""), "{dump}");
            }
        }

        /// The property the ELF path holds, held here too: nothing outside the backends knows
        /// where the allocator put a value, and a guess makes a debugger print a wrong number.
        #[test]
        fn no_parameter_is_given_a_location() {
            for target in ["x86_64_macos", "aarch64_macos"] {
                let Some(dump) = dwarfdump(target, "--debug-info") else {
                    return;
                };
                assert!(!dump.contains("DW_AT_location"), "{target}: {dump}");
                assert!(!dump.contains("DW_TAG_variable"), "{target}: {dump}");
            }
        }

        #[test]
        fn the_line_table_reaches_the_source_lines_of_every_function() {
            let Some(dump) = dwarfdump("x86_64_macos", "--debug-line") else {
                return;
            };
            for line in ["     3 ", "     9 ", "    14 "] {
                assert!(dump.contains(line), "no row for line{line}in {dump}");
            }
        }

        /// Debug info is opt-in: a build without `-g` must be the same object it always was.
        #[test]
        fn a_build_without_debug_lines_is_unchanged() {
            for target in ["x86_64_macos", "aarch64_macos"] {
                let plain = object(target, false);
                assert!(
                    !String::from_utf8_lossy(&plain).contains("__DWARF"),
                    "{target} carries debug sections without -g"
                );
            }
        }
    }

    /// What gdb and lldb can answer about a `-g` build.
    ///
    /// The two disagree about malformed DWARF often enough that passing both is the bar.
    /// Each assertion is a question a person debugging would ask, not a comparison with
    /// another toolchain's output.
    mod debuggers {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        mod native {
            use crate::test_support::TempDir;
            use crate::test_support::lamina_binary;
            use std::{
                env, fs,
                path::{Path, PathBuf},
                process::{self, Command},
            };

            const SOURCE: &str = r#"fn @triple(i64 %n) -> i64 {
          entry:
            %r = mul.i64 %n, 3
            ret.i64 %r
        }

        fn @main() -> i64 {
          entry:
            %v = call @triple(7)
            %d = call @emit(%v)
            ret.i64 0
        }

        fn @emit(i64 %value) -> i64 {
          entry:
            %c = add.i64 %value, 27
            %w = writebyte %c
            ret.i64 0
        }
        "#;

            /// Skips when the tool is missing, unless the suite was told to require it.
            /// The debugger to drive, accepting a version-suffixed spelling: Ubuntu ships
            /// `lldb-18` with no unsuffixed alias unless the `lldb` package is installed.
            fn tool(name: &str) -> Option<String> {
                let runnable = |candidate: &str| {
                    Command::new(candidate)
                        .arg("--version")
                        .output()
                        .is_ok_and(|output| output.status.success())
                };
                if runnable(name) {
                    return Some(name.to_string());
                }
                for suffix in ["-18", "-19", "-17"] {
                    let versioned = format!("{name}{suffix}");
                    if runnable(&versioned) {
                        return Some(versioned);
                    }
                }
                let required =
                    env::var_os("LAMINA_REQUIRE_TOOLS").is_some_and(|value| value == "1");
                assert!(
                    !required,
                    "{name} is not installed (LAMINA_REQUIRE_TOOLS=1)"
                );
                None
            }

            fn build() -> (TempDir, PathBuf) {
                let guard = TempDir::new("lamina-debug-info");
                let stem = format!("lamina-debug-info-{}", process::id());
                let input = guard.path().join(format!("{stem}.lamina"));
                let exe = guard.path().join(&stem);
                fs::write(&input, SOURCE).expect("write program");
                let run = Command::new(lamina_binary())
                    .arg(&input)
                    .arg("-g")
                    .arg("--target")
                    .arg("x86_64_linux")
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

            fn run(tool: &str, arguments: &[&str], exe: &Path) -> String {
                let output = Command::new(tool)
                    .args(arguments)
                    .arg(exe)
                    .output()
                    .unwrap_or_else(|e| panic!("run {tool}: {e}"));
                let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
                text.push_str(&String::from_utf8_lossy(&output.stderr));
                text
            }

            #[test]
            fn gdb_lists_the_functions_with_their_types_and_lines() {
                let Some(gdb) = tool("gdb") else { return };
                let (_guard, exe) = build();
                let text = run(&gdb, &["-batch", "-ex", "info functions"], &exe);

                assert!(text.contains("i64 main(void);"), "{text}");
                // `triple` is on line 1 and `emit` on line 14 of the source above.
                assert!(text.contains("1:\tstatic i64 triple(i64);"), "{text}");
                assert!(text.contains("14:\tstatic i64 emit(i64);"), "{text}");
            }

            #[test]
            fn gdb_names_every_frame_of_a_backtrace() {
                let Some(gdb) = tool("gdb") else { return };
                let (_guard, exe) = build();
                let text = run(
                    &gdb,
                    &["-batch", "-ex", "break emit", "-ex", "run", "-ex", "bt"],
                    &exe,
                );

                assert!(text.contains("in emit ("), "{text}");
                assert!(text.contains("#1  "), "{text}");
                assert!(text.contains("in main ()"), "{text}");
            }

            /// `triple(7)` is 21, and `emit` returns 0 after writing 21 + 27 as a byte. gdb
            /// reads both through `DW_AT_type`, so a wrong base type shows up here.
            #[test]
            fn gdb_reads_a_return_value_through_its_declared_type() {
                let Some(gdb) = tool("gdb") else { return };
                let (_guard, exe) = build();
                let text = run(
                    &gdb,
                    &[
                        "-batch",
                        "-ex",
                        "break triple",
                        "-ex",
                        "run",
                        "-ex",
                        "finish",
                    ],
                    &exe,
                );

                assert!(text.contains("Value returned is $1 = 21"), "{text}");
            }

            /// Nothing knows where the allocator put a parameter, so the answer must be that
            /// it is unavailable. A number here would be a guess a debugger reported as fact.
            #[test]
            fn a_parameter_reads_as_unavailable_rather_than_as_a_guess() {
                let Some(gdb) = tool("gdb") else { return };
                let (_guard, exe) = build();
                let text = run(
                    &gdb,
                    &[
                        "-batch",
                        "-ex",
                        "break triple",
                        "-ex",
                        "run",
                        "-ex",
                        "info args",
                    ],
                    &exe,
                );

                assert!(text.contains("n = <optimized out>"), "{text}");
            }

            #[test]
            fn lldb_stops_at_a_function_and_names_the_caller() {
                let Some(lldb) = tool("lldb") else { return };
                let (_guard, exe) = build();
                let text = run(
                    &lldb,
                    &[
                        "--batch",
                        "-o",
                        "br set -n emit",
                        "-o",
                        "run",
                        "-o",
                        "bt",
                        "-o",
                        "frame variable",
                    ],
                    &exe,
                );

                assert!(text.contains("frame #0:"), "{text}");
                assert!(text.contains("`emit"), "{text}");
                assert!(text.contains("`main"), "{text}");
                assert!(text.contains("value"), "{text}");
            }

            #[test]
            fn the_tree_passes_the_dwarf_verifier() {
                let Some(verifier) = ["llvm-dwarfdump", "llvm-dwarfdump-18", "llvm-dwarfdump-19"]
                    .into_iter()
                    .find(|name| {
                        Command::new(name)
                            .arg("--version")
                            .output()
                            .is_ok_and(|output| output.status.success())
                    })
                else {
                    return;
                };
                let (_guard, exe) = build();
                let text = run(verifier, &["--verify"], &exe);

                assert!(text.contains("No errors."), "{text}");
            }
        }
    }
}
