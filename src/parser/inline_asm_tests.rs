#![allow(clippy::expect_used)]

use super::parse_module;

fn rejected(line: &str) -> String {
    let source = format!("fn @main(i64 %a) -> i64 {{\nentry:\n    {line}\n    ret.i64 0\n}}\n");
    parse_module(&source)
        .expect_err("asm should be rejected")
        .to_string()
}

#[test]
fn a_template_placeholder_must_name_an_operand() {
    for (line, fragment) in [
        ("%v = asm.i64 \"movq $1, $0\", out(rax)", "uses $1"),
        ("%v = asm.i64 \"movq $x, $0\", out(rax)", "stray '$'"),
    ] {
        let message = rejected(line);
        assert!(message.contains(fragment), "{line}\n{message}");
    }
}

#[test]
fn a_register_cannot_be_both_an_operand_and_a_clobber() {
    for line in [
        "asm.void \"nop\", in(rcx) %a, clobber(rcx)",
        // The two spellings name one register, so the width must not hide the clash.
        "asm.void \"nop\", in(rcx) %a, clobber(ecx)",
    ] {
        let message = rejected(line);
        assert!(
            message.contains("both an asm operand and a clobber"),
            "{line}\n{message}"
        );
    }
}

#[test]
fn the_stack_and_frame_pointers_are_off_limits() {
    for (line, fragment) in [
        (
            "asm.void \"nop\", clobber(rsp)",
            "cannot clobber the stack or frame pointer",
        ),
        (
            "asm.void \"nop\", clobber(rbp)",
            "cannot clobber the stack or frame pointer",
        ),
        ("%v = asm.i64 \"nop\", out(rsp)", "as an asm operand"),
    ] {
        let message = rejected(line);
        assert!(message.contains(fragment), "{line}\n{message}");
    }
}

#[test]
fn a_result_needs_exactly_one_out_operand_of_integer_or_pointer_type() {
    for (line, fragment) in [
        ("asm.void \"nop\", out(rax)", "takes no out"),
        ("%v = asm.void \"nop\"", "cannot be bound"),
        ("%v = asm.f64 \"nop\", out(rax)", "integer or ptr"),
        ("%v = asm.i64 \"nop\", in(rcx) %a", "found 0"),
        ("%v = asm.i64 \"nop\", out(rax), out(rcx)", "found 2"),
    ] {
        let message = rejected(line);
        assert!(message.contains(fragment), "{line}\n{message}");
    }
}

#[test]
fn an_input_is_a_value_at_a_location_nothing_else_uses() {
    for (line, fragment) in [
        (
            "%v = asm.i64 \"nop\", in(rcx) %a, in(ecx) %a, out(rax)",
            "used by two asm operands",
        ),
        (
            "%v = asm.i64 \"nop\", in(rcx) 5, out(rax)",
            "must be values",
        ),
    ] {
        let message = rejected(line);
        assert!(message.contains(fragment), "{line}\n{message}");
    }
}

#[test]
fn well_formed_asm_parses() {
    let source = "fn @main(i64 %a) -> i64 {\nentry:\n    %v = asm.i64 \"movq $$7, $0\\nmovq $0, $0\", in(rcx) %a, out(rax), clobber(rdx, memory, flags)\n    asm.void \"nop\", clobber(rbx)\n    ret.i64 %v\n}\n";
    parse_module(source).expect("well-formed asm should parse");
}
