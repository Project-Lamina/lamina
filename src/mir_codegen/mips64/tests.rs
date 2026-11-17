use super::abi::{
    ArgKind, ArgPlace, call_area, classify_kinds, frame_bytes, incoming_stack_offset,
};
use crate::mir::{Aggregate, AggregateClass, AggregateShape, MirType, ScalarType};

fn scalars(types: &[MirType]) -> Vec<ArgKind> {
    types.iter().copied().map(ArgKind::scalar).collect()
}

#[test]
fn argument_registers_are_numbered_by_doubleword_across_both_files() {
    let types = [
        MirType::Scalar(ScalarType::I64),
        MirType::Scalar(ScalarType::F64),
        MirType::Scalar(ScalarType::I64),
        MirType::Scalar(ScalarType::F32),
    ];
    assert_eq!(
        classify_kinds(&scalars(&types), 0),
        vec![
            ArgPlace::Gpr(0),
            ArgPlace::Fpr(1),
            ArgPlace::Gpr(2),
            ArgPlace::Fpr(3)
        ]
    );
}

#[test]
fn the_ninth_argument_goes_to_the_stack() {
    let types = vec![MirType::Scalar(ScalarType::I64); 10];
    let kinds = scalars(&types);
    let places = classify_kinds(&kinds, 0);
    assert_eq!(places[7], ArgPlace::Gpr(7));
    assert_eq!(places[8], ArgPlace::Stack(0));
    assert_eq!(places[9], ArgPlace::Stack(1));
    assert_eq!(call_area(&kinds, 0), 16);
}

#[test]
fn an_aggregate_takes_consecutive_doublewords_and_splits_at_the_eighth() {
    let aggregate = Aggregate {
        size: 24,
        align: 8,
        class: AggregateClass::Integer,
        shape: AggregateShape::Composite,
    };
    let kinds = vec![
        ArgKind::scalar(MirType::Scalar(ScalarType::I64)),
        ArgKind::scalar(MirType::Scalar(ScalarType::I64)),
        ArgKind::scalar(MirType::Scalar(ScalarType::I64)),
        ArgKind::scalar(MirType::Scalar(ScalarType::I64)),
        ArgKind::scalar(MirType::Scalar(ScalarType::I64)),
        ArgKind::scalar(MirType::Scalar(ScalarType::I64)),
        ArgKind::scalar(MirType::Scalar(ScalarType::I64)),
        ArgKind {
            ty: MirType::Scalar(ScalarType::Ptr),
            aggregate: Some(aggregate),
        },
    ];
    assert_eq!(
        classify_kinds(&kinds, 0)[7],
        ArgPlace::Split { first: 7, words: 3 }
    );
    assert_eq!(call_area(&kinds, 0), 16);
}

#[test]
fn a_hidden_result_pointer_shifts_every_argument_by_one_doubleword() {
    let types = vec![MirType::Scalar(ScalarType::I64); 2];
    assert_eq!(
        classify_kinds(&scalars(&types), 1),
        vec![ArgPlace::Gpr(1), ArgPlace::Gpr(2)]
    );
}

#[test]
fn the_frame_keeps_the_sixteen_byte_stack_alignment() {
    for locals in 0..40 {
        for outgoing in 0..40 {
            assert_eq!(frame_bytes(locals, outgoing) % 16, 0, "{locals} {outgoing}");
        }
    }
}

#[test]
fn incoming_stack_arguments_start_above_the_saved_registers() {
    assert_eq!(incoming_stack_offset(0), 16);
    assert_eq!(incoming_stack_offset(1), 24);
}

fn module_calling_external(ty: MirType) -> crate::mir::Module {
    use crate::mir::{FunctionBuilder, Instruction, Module, Operand, Register, VirtualReg};
    let reg = Register::Virtual(VirtualReg::gpr(0));
    let caller = FunctionBuilder::new("caller")
        .returns(ty)
        .block("entry")
        .instr(Instruction::Call {
            name: "ext".to_string(),
            args: Vec::new(),
            ret: Some(reg.clone()),
            ret_ty: Some(ty),
        })
        .instr(Instruction::Ret {
            value: Some(Operand::Register(reg)),
        })
        .build();
    let mut module = Module::new("narrow_result");
    module.add_function(caller);
    module
}

#[test]
fn narrow_external_results_are_widened_after_the_call() {
    use crate::mir::{MirType, ScalarType};
    use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
    for (ty, extend) in [
        (ScalarType::I1, Some("andi $2, $2, 1")),
        (ScalarType::I8, Some("dsll $2, $2, 56\n    dsra $2, $2, 56")),
        (
            ScalarType::I16,
            Some("dsll $2, $2, 48\n    dsra $2, $2, 48"),
        ),
        (ScalarType::I32, None),
    ] {
        let module = module_calling_external(MirType::Scalar(ty));
        let mut out = Vec::new();
        super::generate_mir_mips64_with_units_and_settings(
            &module,
            &mut out,
            TargetOperatingSystem::Linux,
            1,
            &crate::mir_codegen::settings::MirCodegenSettings::default(),
            TargetArchitecture::Mips64,
        )
        .expect("codegen");
        let assembly = String::from_utf8(out).expect("utf8");
        let after_call = assembly.split("jalr").nth(1).expect("call emitted");
        let after_delay_slot = after_call.split_once("nop\n").expect("delay slot").1;
        match extend {
            Some(instructions) => assert!(
                after_delay_slot.starts_with(&format!("    {instructions}\n")),
                "{assembly}"
            ),
            // The N64 ABI keeps a 32-bit return sign-extended, so no instruction is owed.
            None => assert!(!after_delay_slot.starts_with("    dsll"), "{assembly}"),
        }
    }
}
