//! Mach-O `__LD,__compact_unwind` entries.
//!
//! Darwin's unwinder looks a return address up in `__unwind_info`, which the linker builds
//! from the 32-byte entries in this section: function address, length, a packed encoding
//! of the frame shape, personality and LSDA. The encoding has room for one of a handful of
//! shapes and nothing else, so a frame it cannot name exactly gets no entry at all. An
//! absent entry stops an unwind; a wrong one resumes at a wrong address.

use std::io::{self, Write};

use lamina_platform::TargetOperatingSystem;

use lamina_codegen::os_uses_macho;

const UNWIND_X86_64_MODE_RBP_FRAME: u32 = 0x0100_0000;
const UNWIND_ARM64_MODE_FRAME: u32 = 0x0400_0000;

/// The five slots the x86_64 frame encoding has for callee-saved registers.
const X86_64_FRAME_REGISTERS: usize = 5;

/// `UNWIND_X86_64_RBP_FRAME_OFFSET` is eight bits of word count below `rbp`.
const X86_64_FRAME_OFFSET_MAX: usize = 0xff;

/// The frame record `stp x29, x30, [sp, #-16]!` the arm64 frame encoding assumes.
const AARCH64_FRAME_RECORD_BYTES: i32 = 16;

/// The register numbers the x86_64 frame encoding can name. `rbp` is implied by the mode
/// and has no slot of its own.
fn x86_64_register_number(name: &str) -> Option<u32> {
    Some(match name {
        "rbx" => 1,
        "r12" => 2,
        "r13" => 3,
        "r14" => 4,
        "r15" => 5,
        _ => return None,
    })
}

/// The compact encoding for an x86_64 frame, or `None` when the shape has no exact one.
///
/// The prologue is `push rbp`, `mov rsp, rbp`, one stack allocation, then a `mov` per
/// callee-saved register. `UNWIND_X86_64_MODE_RBP_FRAME` restores `rsp` from `rbp`, so the
/// allocation needs no description and its size does not matter. What it does constrain is
/// the saved registers: five slots, read upwards from `rbp - 8 * offset`, each naming one
/// of `rbx` and `r12` through `r15`. A function that saves more than five, saves one the
/// encoding cannot name, or puts the lowest slot further than 255 words below `rbp`, gets
/// no entry.
pub fn plan_x86_64(
    target_os: TargetOperatingSystem,
    naked: bool,
    has_frame: bool,
    saved_regs: &[&str],
    total_slots: usize,
) -> Option<u32> {
    if !os_uses_macho(target_os) || naked || !has_frame {
        return None;
    }
    if saved_regs.is_empty() {
        return Some(UNWIND_X86_64_MODE_RBP_FRAME);
    }
    if saved_regs.len() > X86_64_FRAME_REGISTERS {
        return None;
    }
    // Register `index` sits at `rbp - 8 * (total_slots + index + 1)`, so the deepest is the
    // last one, and the encoding reads upwards from there.
    let offset = total_slots + saved_regs.len();
    if offset > X86_64_FRAME_OFFSET_MAX {
        return None;
    }
    let mut registers = 0u32;
    for (slot, name) in saved_regs.iter().rev().enumerate() {
        registers |= x86_64_register_number(name)? << (3 * slot);
    }
    Some(UNWIND_X86_64_MODE_RBP_FRAME | (offset as u32) << 16 | registers)
}

/// The compact encoding for an AArch64 frame, or `None` when the shape has no exact one.
///
/// `UNWIND_ARM64_MODE_FRAME` describes the standard frame record and, optionally, register
/// pairs stored in a fixed order immediately below `x29`: `x19`/`x20` at `x29 - 8` and
/// `x29 - 16`, then `x21`/`x22`, and so on. This backend stores its callee-saved registers
/// at the bottom of the frame instead, ascending, and not in pairs, so a function that
/// saves any of them cannot be described and gets no entry. A grown frame record, which
/// only a Windows variadic function has, is not the shape the encoding assumes either.
pub fn plan_aarch64(
    target_os: TargetOperatingSystem,
    frame_record: i32,
    callee_saved: &[&'static str],
) -> Option<u32> {
    let describable = os_uses_macho(target_os)
        && callee_saved.is_empty()
        && frame_record == AARCH64_FRAME_RECORD_BYTES;
    describable.then_some(UNWIND_ARM64_MODE_FRAME)
}

/// The end of a function's instructions, which the assembler subtracts to get its length.
/// Mach-O keeps an `L`-prefixed local out of the symbol table.
pub fn function_end_label(label: &str) -> String {
    format!("Lcompact_unwind_end_{label}")
}

/// Emit one entry, then return to `.text` for whatever follows.
pub fn emit<W: Write>(writer: &mut W, label: &str, encoding: u32) -> io::Result<()> {
    writeln!(writer, ".section __LD,__compact_unwind,regular,debug")?;
    writeln!(writer, "    .p2align 3")?;
    writeln!(writer, "    .quad {label}")?;
    writeln!(writer, "    .long {}-{label}", function_end_label(label))?;
    writeln!(writer, "    .long {encoding}")?;
    writeln!(writer, "    .quad 0")?;
    writeln!(writer, "    .quad 0")?;
    writeln!(writer, ".text")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MACOS: TargetOperatingSystem = TargetOperatingSystem::MacOS;

    #[test]
    fn a_frameless_or_naked_x86_64_function_gets_no_entry() {
        assert_eq!(plan_x86_64(MACOS, false, false, &[], 0), None);
        assert_eq!(plan_x86_64(MACOS, true, true, &[], 0), None);
    }

    #[test]
    fn no_entry_is_planned_off_mach_o() {
        for os in [
            TargetOperatingSystem::Linux,
            TargetOperatingSystem::Windows,
            TargetOperatingSystem::FreeBSD,
        ] {
            assert_eq!(plan_x86_64(os, false, true, &[], 0), None);
            assert_eq!(plan_aarch64(os, 16, &[]), None);
        }
    }

    #[test]
    fn a_plain_x86_64_frame_is_the_bare_mode() {
        assert_eq!(
            plan_x86_64(MACOS, false, true, &[], 9),
            Some(UNWIND_X86_64_MODE_RBP_FRAME)
        );
    }

    #[test]
    fn x86_64_saved_registers_read_upwards_from_the_deepest_slot() {
        // rbx at rbp-24, r12 at rbp-32: the deepest is four words down, and the slots run
        // r12 then rbx from there.
        let encoding = plan_x86_64(MACOS, false, true, &["rbx", "r12"], 2).expect("described");
        assert_eq!(encoding & 0x0f00_0000, UNWIND_X86_64_MODE_RBP_FRAME);
        assert_eq!(encoding >> 16 & 0xff, 4, "frame offset in words");
        assert_eq!(encoding & 0x7fff, 2 | 1 << 3, "r12 then rbx");
    }

    #[test]
    fn an_x86_64_register_the_encoding_cannot_name_gets_no_entry() {
        assert_eq!(plan_x86_64(MACOS, false, true, &["rsi"], 0), None);
    }

    #[test]
    fn more_than_five_x86_64_saved_registers_gets_no_entry() {
        let saved = ["rbx", "r12", "r13", "r14", "r15", "rbx"];
        assert_eq!(plan_x86_64(MACOS, false, true, &saved, 0), None);
        assert!(plan_x86_64(MACOS, false, true, &saved[..5], 0).is_some());
    }

    #[test]
    fn an_x86_64_slot_past_the_offset_field_gets_no_entry() {
        assert_eq!(
            plan_x86_64(MACOS, false, true, &["rbx"], X86_64_FRAME_OFFSET_MAX),
            None
        );
        assert!(plan_x86_64(MACOS, false, true, &["rbx"], X86_64_FRAME_OFFSET_MAX - 1).is_some());
    }

    #[test]
    fn an_aarch64_frame_record_alone_is_describable() {
        assert_eq!(plan_aarch64(MACOS, 16, &[]), Some(UNWIND_ARM64_MODE_FRAME));
    }

    #[test]
    fn aarch64_callee_saved_registers_or_a_grown_record_get_no_entry() {
        assert_eq!(plan_aarch64(MACOS, 16, &["x19"]), None);
        assert_eq!(plan_aarch64(MACOS, 80, &[]), None);
    }

    #[test]
    fn an_entry_is_thirty_two_bytes_of_address_length_encoding_and_two_nulls() {
        let mut out = Vec::new();
        emit(&mut out, "_f", UNWIND_ARM64_MODE_FRAME).expect("emit");
        let asm = String::from_utf8(out).expect("utf8");
        assert!(asm.contains("\n    .quad _f\n"), "{asm}");
        assert!(
            asm.contains("\n    .long Lcompact_unwind_end__f-_f\n    .long 67108864\n"),
            "{asm}"
        );
        assert!(asm.ends_with("    .quad 0\n    .quad 0\n.text\n"), "{asm}");
    }

    use crate::api::{
        ObjectBuild, Target, compile_module_to_object, compile_module_to_object_with,
    };
    use crate::parser::parse_module;
    use crate::test_support::macho::MachO;
    use std::str::FromStr;

    const ENTRY_BYTES: usize = 32;
    const UNWIND_X86_64_MODE_RBP_FRAME: u32 = 0x0100_0000;
    const UNWIND_ARM64_MODE_FRAME: u32 = 0x0400_0000;
    const S_ATTR_DEBUG: u32 = 0x0200_0000;
    const RELOC_UNSIGNED: u32 = 0;

    const CALLS_IR: &str = r#"
    fn @callee(i64 %a) -> i64 {
    entry:
        %r = add.i64 %a, 1
        ret.i64 %r
    }

    fn @main() -> i64 {
    entry:
        %x = call @callee(41)
        ret.i64 %x
    }
    "#;

    fn object(target: &str, ir: &str) -> Vec<u8> {
        let target = Target::from_str(target).expect("target string parses");
        let module = parse_module(ir).expect("module parses");
        compile_module_to_object(&module, target).expect("object bytes")
    }

    /// Register pressure, which is what puts a callee-saved register in the frame at all,
    /// only arises once the global allocator runs.
    fn optimized_object(target: &str, ir: &str) -> Vec<u8> {
        let target = Target::from_str(target).expect("target string parses");
        let module = parse_module(ir).expect("module parses");
        let build = ObjectBuild {
            opt_level: 2,
            ..ObjectBuild::default()
        };
        compile_module_to_object_with(&module, target, &build).expect("object bytes")
    }

    const FAR_FRAME_IR: &str = r#"
    fn @farframe(i64 %seed) -> i64 {
    entry:
      %buf = stackalloc.[4096 x i64]
      %i = stackalloc.i64
      store.i64 %i, 0
      jmp fill

    fill:
      %iv = load.i64 %i
      %go = lt.i64 %iv, 4096
      br %go, fill_body, done

    fill_body:
      %slot = getelem.i64 %buf, %iv
      %v1 = add.i64 %iv, %seed
      %v2 = mul.i64 %v1, 3
      %v3 = xor.i64 %v2, %v1
      %v4 = mul.i64 %v3, 5
      %v5 = xor.i64 %v4, %v2
      %v6 = mul.i64 %v5, 7
      %v7 = xor.i64 %v6, %v3
      %v8 = add.i64 %v7, %v4
      %v9 = xor.i64 %v8, %v5
      %v10 = add.i64 %v9, %v6
      store.i64 %slot, %v10
      %in = add.i64 %iv, 1
      store.i64 %i, %in
      jmp fill

    done:
      %last = getelem.i64 %buf, 4095
      %out = load.i64 %last
      ret.i64 %out
    }
    "#;

    struct Entry {
        function: String,
        length: u32,
        encoding: u32,
    }

    fn entries(object: &MachO) -> Vec<Entry> {
        let Some(section) = object.find_section("__LD", "__compact_unwind") else {
            return Vec::new();
        };
        assert_eq!(
            section.flags & S_ATTR_DEBUG,
            S_ATTR_DEBUG,
            "the section is marked debug so the linker consumes it rather than copying it"
        );
        assert_eq!(section.align, 3, "entries are eight-byte aligned");
        assert_eq!(
            section.size as usize % ENTRY_BYTES,
            0,
            "the section is a whole number of entries"
        );
        let count = section.size as usize / ENTRY_BYTES;
        (0..count)
            .map(|index| {
                let at = index * ENTRY_BYTES;
                let reloc = section
                    .relocs
                    .iter()
                    .find(|r| r.address as usize == at)
                    .expect("the function address is a relocation");
                assert_eq!(reloc.kind, RELOC_UNSIGNED);
                assert_eq!(reloc.length, 3, "an eight-byte pointer");
                assert!(!reloc.pcrel);
                for field in [at + 16, at + 24] {
                    assert!(
                        !section.relocs.iter().any(|r| r.address as usize == field),
                        "personality and LSDA are null, not relocated"
                    );
                    assert_eq!(
                        u64::from_le_bytes(
                            section.data[field..field + 8]
                                .try_into()
                                .expect("eight bytes")
                        ),
                        0
                    );
                }
                Entry {
                    function: object.symbol_name(reloc.symbolnum).to_string(),
                    length: u32::from_le_bytes(
                        section.data[at + 8..at + 12]
                            .try_into()
                            .expect("four bytes"),
                    ),
                    encoding: u32::from_le_bytes(
                        section.data[at + 12..at + 16]
                            .try_into()
                            .expect("four bytes"),
                    ),
                }
            })
            .collect()
    }

    /// A length of zero would make the linker's range for the function empty, so every entry
    /// has to carry the measured distance to the end of its instructions.
    fn assert_lengths_cover_the_text(object: &MachO, entries: &[Entry]) {
        let text = object.section("__TEXT", "__text");
        for entry in entries {
            let symbol = object.symbol(&entry.function);
            let start = symbol.n_value - text.addr;
            assert!(entry.length > 0, "{} has a length", entry.function);
            assert!(
                start + u64::from(entry.length) <= text.size,
                "{} runs past __text",
                entry.function
            );
        }
    }

    #[test]
    fn x86_64_frames_are_described_as_rbp_frames() {
        let object = MachO::parse(&object("x86_64_macos", CALLS_IR));
        let entries = entries(&object);
        assert_lengths_cover_the_text(&object, &entries);
        let main = entries
            .iter()
            .find(|e| e.function == "_main")
            .expect("_main has a frame and an entry");
        assert_eq!(
            main.encoding & 0x0f00_0000,
            UNWIND_X86_64_MODE_RBP_FRAME,
            "mode"
        );
        // `_callee` is a leaf with no frame, so it is described by nothing at all.
        assert!(
            !entries.iter().any(|e| e.function == "_callee"),
            "a frameless function gets no entry"
        );
    }

    #[test]
    fn an_x86_64_saved_register_is_named_at_its_own_slot() {
        let object = MachO::parse(&object("x86_64_macos", CALLS_IR));
        let main = entries(&object)
            .into_iter()
            .find(|e| e.function == "_main")
            .expect("_main entry");
        let offset = main.encoding >> 16 & 0xff;
        let registers = main.encoding & 0x7fff;
        assert_eq!(
            (registers != 0),
            (offset != 0),
            "a named register needs a frame offset and the other way round"
        );
        if registers != 0 {
            assert_eq!(registers & 0x7, 1, "the deepest slot holds rbx");
            assert_eq!(offset, 1, "rbx sits one word below rbp");
        }
    }

    #[test]
    fn aarch64_frames_are_described_as_standard_frame_records() {
        let object = MachO::parse(&object("aarch64_macos", CALLS_IR));
        for entry in entries(&object) {
            assert_eq!(
                entry.encoding, UNWIND_ARM64_MODE_FRAME,
                "{} is a bare frame record",
                entry.function
            );
        }
    }

    /// The frame size itself never disqualifies an `rbp` frame: the mode restores `rsp` from
    /// `rbp`, so the allocation needs no description.
    #[test]
    fn a_deep_frame_with_nothing_saved_is_still_described() {
        const DEEP: &str = r#"
    fn @deep(i64 %seed) -> i64 {
    entry:
        %buf = stackalloc.[4096 x i64]
        %slot = getelem.i64 %buf, 7
        store.i64 %slot, %seed
        %v = load.i64 %slot
        ret.i64 %v
    }
    "#;
        let object = MachO::parse(&object("x86_64_macos", DEEP));
        let deep = entries(&object)
            .into_iter()
            .find(|e| e.function == "_deep")
            .expect("_deep entry");
        assert_eq!(deep.encoding, UNWIND_X86_64_MODE_RBP_FRAME, "nothing saved");
    }

    /// A saved register further below `rbp` than the eight-bit offset field reaches cannot be
    /// named, so the function is left out rather than described approximately.
    #[test]
    fn a_saved_register_out_of_reach_of_the_offset_field_gets_no_entry() {
        let object = MachO::parse(&optimized_object("x86_64_macos", FAR_FRAME_IR));
        assert!(
            !entries(&object).iter().any(|e| e.function == "_farframe"),
            "the deep frame with saved registers is left out"
        );
    }

    #[test]
    fn no_compact_unwind_section_is_emitted_off_mach_o() {
        for target in ["x86_64_linux", "aarch64_linux", "x86_64_windows"] {
            let object = object(target, CALLS_IR);
            assert!(
                !object.windows(16).any(|w| w == b"__compact_unwind"),
                "{target} carries no compact unwind section"
            );
        }
    }
}
