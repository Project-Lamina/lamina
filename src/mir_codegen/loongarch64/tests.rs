use super::abi::{ArgPlace, classify, load_imm, outgoing_bytes};
use crate::mir::MirType;
use lamina_mir::ScalarType;

/// Executes the `addi.d`, `lu12i.w`, `ori`, `lu32i.d` and `lu52i.d` lines `load_imm`
/// writes, using the architectural definition of each.
fn run_load_imm(value: i64) -> i64 {
    let mut out = Vec::new();
    load_imm(&mut out, "$t0", value).expect("write to vec");
    let mut reg = 0i64;
    for line in String::from_utf8(out).expect("utf8").lines() {
        let line = line.trim();
        let (op, args) = line.split_once(' ').expect("mnemonic and operands");
        let args: Vec<&str> = args.split(", ").collect();
        let imm = |s: &str| s.parse::<i64>().expect("immediate");
        reg = match (op, args.as_slice()) {
            ("addi.d", ["$t0", "$zero", v]) => imm(v),
            ("lu12i.w", ["$t0", v]) => (imm(v) << 12) as i32 as i64,
            ("ori", ["$t0", "$t0", v]) => reg | imm(v),
            ("lu32i.d", ["$t0", v]) => (reg & !(0xf_ffff << 32)) | ((imm(v) & 0xf_ffff) << 32),
            ("lu52i.d", ["$t0", "$t0", v]) => (reg & 0xf_ffff_ffff_ffff) | ((imm(v) & 0xfff) << 52),
            other => panic!("unexpected instruction {other:?}"),
        };
    }
    reg
}

#[test]
fn load_imm_reconstructs_every_boundary_value() {
    let values = [
        0,
        1,
        -1,
        2047,
        -2048,
        2048,
        -2049,
        4095,
        4096,
        -4097,
        i64::from(i32::MAX),
        i64::from(i32::MIN),
        i64::from(i32::MAX) + 1,
        i64::from(i32::MIN) - 1,
        0x1234_5678_9abc_def0,
        i64::MAX,
        i64::MIN,
        0x0000_0001_0000_0000,
        -0x0000_0001_0000_0000,
    ];
    for value in values {
        assert_eq!(run_load_imm(value), value, "load_imm({value:#x})");
    }
}

#[test]
fn integer_arguments_past_eight_go_to_the_stack() {
    let types: Vec<MirType> = (0..10).map(|_| MirType::Scalar(ScalarType::I64)).collect();
    let places = classify(&types);
    assert_eq!(places[7], ArgPlace::Gpr(7));
    assert_eq!(places[8], ArgPlace::Stack(0));
    assert_eq!(places[9], ArgPlace::Stack(1));
    assert_eq!(outgoing_bytes(&places), 16);
}

#[test]
fn floats_take_their_own_bank() {
    let types = [
        MirType::Scalar(ScalarType::I64),
        MirType::Scalar(ScalarType::F64),
        MirType::Scalar(ScalarType::I64),
        MirType::Scalar(ScalarType::F32),
    ];
    let places = classify(&types);
    assert_eq!(
        places,
        [
            ArgPlace::Gpr(0),
            ArgPlace::Fpr(0),
            ArgPlace::Gpr(1),
            ArgPlace::Fpr(1),
        ]
    );
    assert_eq!(outgoing_bytes(&places), 0);
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
    use crate::mir::MirType;
    use lamina_platform::TargetOperatingSystem;
    for (ty, extend) in [
        (ScalarType::I1, Some("andi $a0, $a0, 1")),
        (
            ScalarType::I8,
            Some("slli.d $a0, $a0, 56\n    srai.d $a0, $a0, 56"),
        ),
        (
            ScalarType::I16,
            Some("slli.d $a0, $a0, 48\n    srai.d $a0, $a0, 48"),
        ),
        (ScalarType::I32, None),
    ] {
        let module = module_calling_external(MirType::Scalar(ty));
        let mut out = Vec::new();
        super::generate_mir_loongarch64_with_units_and_settings(
            &module,
            &mut out,
            TargetOperatingSystem::Linux,
            1,
            &crate::mir_codegen::settings::MirCodegenSettings::default(),
        )
        .expect("codegen");
        let assembly = String::from_utf8(out).expect("utf8");
        let after_call = assembly.split("bl ext").nth(1).expect("call emitted");
        match extend {
            Some(instructions) => assert!(
                after_call.starts_with(&format!("\n    {instructions}\n")),
                "{assembly}"
            ),
            // The LP64 ABI keeps a 32-bit return sign-extended, so no instruction is owed.
            None => assert!(!after_call.contains("slli.d $a0, $a0, 32"), "{assembly}"),
        }
    }
}
