//! Line-level cleanups over the finished assembly of one x86_64 function.
//!
//! Instruction selection works one MIR instruction at a time and cannot see that the
//! neighbour it just emitted already did the work. These rules run once over the text a
//! function produced, where that neighbour is visible.

use super::moves::is_self_move;

/// The mnemonic and the operand text of an instruction line. Labels, directives and
/// blank lines are not instructions.
fn parts(line: &str) -> Option<(&str, &str)> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('.') || line.starts_with('#') || line.ends_with(':') {
        return None;
    }
    match line.split_once(char::is_whitespace) {
        Some((mnemonic, operands)) => Some((mnemonic, operands.trim())),
        None => Some((line, "")),
    }
}

/// The destination operand, which is the last one in AT&T order.
fn destination(operands: &str) -> &str {
    match operands.rsplit_once(", ") {
        Some((_, destination)) => destination,
        None => operands,
    }
}

/// Mnemonics that leave ZF, SF and PF describing their own result, so a compare of that
/// result against zero would set them the same way. Multiplies and `lea` are absent
/// because they do not, and shifts because a count of zero leaves the flags alone.
fn writes_result_flags(mnemonic: &str) -> bool {
    matches!(
        mnemonic.get(..mnemonic.len().saturating_sub(1)),
        Some("add" | "sub" | "and" | "or" | "xor" | "neg" | "inc" | "dec")
    )
}

/// Conditional jumps that read ZF, SF or PF alone. A compare against zero also clears CF
/// and OF, which the arithmetic it replaces does not, so jumps reading those stay.
fn reads_result_flags_only(mnemonic: &str) -> bool {
    matches!(
        mnemonic,
        "je" | "jz" | "jne" | "jnz" | "js" | "jns" | "jp" | "jnp" | "jpe" | "jpo"
    )
}

/// Anything that reads the flags at all.
fn reads_flags(mnemonic: &str) -> bool {
    if mnemonic.starts_with("set")
        || mnemonic.starts_with("cmov")
        || mnemonic.starts_with("lahf")
        || mnemonic.starts_with("pushf")
    {
        return true;
    }
    if mnemonic.starts_with('j') {
        return !mnemonic.starts_with("jmp");
    }
    matches!(mnemonic.get(..3), Some("adc" | "sbb" | "rcl" | "rcr"))
}

/// `cmpX $0, <dst>`, as the width suffix and the destination it tests.
fn zero_compare(line: &str) -> Option<(u8, &str)> {
    let (mnemonic, operands) = parts(line)?;
    let suffix = mnemonic.strip_prefix("cmp")?;
    let &[width] = suffix.as_bytes() else {
        return None;
    };
    Some((width, operands.strip_prefix("$0, ")?))
}

/// True when `line` is arithmetic of the same width that already left the flags
/// describing `target`.
fn settles_flags_for(line: &str, width: u8, target: &str) -> bool {
    let Some((mnemonic, operands)) = parts(line) else {
        return false;
    };
    writes_result_flags(mnemonic)
        && mnemonic.as_bytes().last() == Some(&width)
        && destination(operands) == target
}

/// The mnemonic of the first instruction at or after `from`.
fn mnemonic_at_or_after<'a>(lines: &[&'a str], from: usize) -> Option<&'a str> {
    lines
        .get(from..)?
        .iter()
        .find_map(|line| parts(line).map(|(mnemonic, _)| mnemonic))
}

/// The mnemonic of the first instruction of the block `label` names, or `None` when the
/// label is not defined in this function.
fn mnemonic_at_label<'a>(lines: &[&'a str], label: &str) -> Option<&'a str> {
    let target = lines
        .iter()
        .position(|line| line.trim().strip_suffix(':') == Some(label))?;
    mnemonic_at_or_after(lines, target + 1)
}

/// Whether the compare on `index` only repeats what the line before it already did.
fn compare_is_redundant(lines: &[&str], keep: &[bool], index: usize) -> bool {
    let Some((width, target)) = zero_compare(lines[index]) else {
        return false;
    };
    let Some(previous) = index.checked_sub(1) else {
        return false;
    };
    if !keep[previous] || !settles_flags_for(lines[previous], width, target) {
        return false;
    }
    // The jump that follows has to be the only reader left, on both of its paths.
    let Some((jump, operands)) = lines.get(index + 1).and_then(|line| parts(line)) else {
        return false;
    };
    if !reads_result_flags_only(jump) {
        return false;
    }
    let fallthrough = mnemonic_at_or_after(lines, index + 2);
    if fallthrough.is_none_or(reads_flags) {
        return false;
    }
    match mnemonic_at_label(lines, operands) {
        Some(taken) => !reads_flags(taken),
        None => false,
    }
}

/// Apply every rule to one function's assembly.
pub(super) fn run(asm: Vec<u8>) -> Vec<u8> {
    let Ok(text) = String::from_utf8(asm) else {
        return Vec::new();
    };
    let lines: Vec<&str> = text.lines().collect();
    let mut keep = vec![true; lines.len()];
    for index in 0..lines.len() {
        if is_self_move(lines[index]) || compare_is_redundant(&lines, &keep, index) {
            keep[index] = false;
        }
    }
    let mut out = String::with_capacity(text.len());
    for (line, &keep) in lines.iter().zip(&keep) {
        if keep {
            out.push_str(line);
            out.push('\n');
        }
    }
    out.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_on(lines: &[&str]) -> Vec<String> {
        let asm = lines.join("\n") + "\n";
        String::from_utf8(run(asm.into_bytes()))
            .expect("peephole output is utf-8")
            .lines()
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn subtraction_already_settled_the_flags_the_compare_tests() {
        let out = run_on(&[
            "f:",
            "    subq $2, %r9",
            "    cmpq $0, %r9",
            "    jne .L_body",
            "    jmp .L_done",
            ".L_body:",
            "    addq %rax, %rcx",
            "    jmp f",
            ".L_done:",
            "    ret",
        ]);
        assert!(
            !out.iter().any(|line| line.contains("cmpq")),
            "compare kept: {out:?}"
        );
        assert!(out.iter().any(|line| line.contains("subq $2, %r9")));
        assert!(out.iter().any(|line| line.contains("jne .L_body")));
    }

    #[test]
    fn a_signed_jump_still_needs_its_own_compare() {
        // `subq` leaves OF set from the subtraction; `cmpq $0` would clear it, and `jl`
        // reads it.
        let out = run_on(&[
            "f:",
            "    subq $2, %r9",
            "    cmpq $0, %r9",
            "    jl .L_body",
            "    jmp .L_done",
            ".L_body:",
            "    ret",
            ".L_done:",
            "    ret",
        ]);
        assert!(out.iter().any(|line| line.contains("cmpq $0, %r9")));
    }

    #[test]
    fn a_multiply_does_not_settle_the_flags() {
        let out = run_on(&[
            "f:",
            "    imulq $3, %r9",
            "    cmpq $0, %r9",
            "    jne .L_body",
            "    jmp .L_done",
            ".L_body:",
            "    ret",
            ".L_done:",
            "    ret",
        ]);
        assert!(out.iter().any(|line| line.contains("cmpq $0, %r9")));
    }

    #[test]
    fn a_compare_of_another_register_stays() {
        let out = run_on(&[
            "f:",
            "    subq $2, %r9",
            "    cmpq $0, %rax",
            "    jne .L_body",
            "    jmp .L_done",
            ".L_body:",
            "    ret",
            ".L_done:",
            "    ret",
        ]);
        assert!(out.iter().any(|line| line.contains("cmpq $0, %rax")));
    }

    #[test]
    fn a_second_reader_of_the_flags_keeps_the_compare() {
        let out = run_on(&[
            "f:",
            "    subq $2, %r9",
            "    cmpq $0, %r9",
            "    jne .L_body",
            "    setg %al",
            ".L_body:",
            "    ret",
        ]);
        assert!(out.iter().any(|line| line.contains("cmpq $0, %r9")));
    }

    #[test]
    fn a_reader_at_the_jump_target_keeps_the_compare() {
        let out = run_on(&[
            "f:",
            "    subq $2, %r9",
            "    cmpq $0, %r9",
            "    jne .L_body",
            "    jmp .L_done",
            ".L_body:",
            "    cmovaq %rax, %rcx",
            "    ret",
            ".L_done:",
            "    ret",
        ]);
        assert!(out.iter().any(|line| line.contains("cmpq $0, %r9")));
    }

    #[test]
    fn a_jump_out_of_the_function_keeps_the_compare() {
        let out = run_on(&[
            "f:",
            "    subq $2, %r9",
            "    cmpq $0, %r9",
            "    jne elsewhere",
            "    ret",
        ]);
        assert!(out.iter().any(|line| line.contains("cmpq $0, %r9")));
    }

    #[test]
    fn widths_have_to_agree() {
        let out = run_on(&[
            "f:",
            "    addl $1, %eax",
            "    cmpq $0, %rax",
            "    jne .L_body",
            "    jmp .L_done",
            ".L_body:",
            "    ret",
            ".L_done:",
            "    ret",
        ]);
        assert!(out.iter().any(|line| line.contains("cmpq $0, %rax")));
    }

    #[test]
    fn self_moves_still_go() {
        let out = run_on(&["f:", "    movq %rax, %rax", "    ret"]);
        assert!(!out.iter().any(|line| line.contains("movq")));
    }
}
