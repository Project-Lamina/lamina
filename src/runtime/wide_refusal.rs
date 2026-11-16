//! Finds the 128-bit integer types an IR module uses, by walking its structure.

use lamina_ir::{
    function::Function,
    instruction::{AsmOperand, Instruction},
    module::{GlobalInit, InitElement, Module},
    types::{Literal, PrimitiveType, Type, Value},
};
use std::collections::HashSet;

/// The first 128-bit integer type the module uses, as its IR spelling.
pub(crate) fn first_wide_integer(module: &Module<'_>) -> Option<&'static str> {
    let mut walk = Walk {
        module,
        named_seen: HashSet::new(),
    };
    walk.module()
}

fn wide_primitive(ty: PrimitiveType) -> Option<&'static str> {
    match ty {
        PrimitiveType::I128 => Some("i128"),
        PrimitiveType::U128 => Some("u128"),
        _ => None,
    }
}

struct Walk<'m, 'a> {
    module: &'m Module<'a>,
    named_seen: HashSet<&'a str>,
}

type Found = Option<&'static str>;

/// Returns the wide type from the enclosing walk step when `found` names one.
macro_rules! hit {
    ($found:expr) => {
        if let Some(wide) = $found {
            return Some(wide);
        }
    };
}

impl<'a> Walk<'_, 'a> {
    fn module(&mut self) -> Found {
        let module = self.module;
        for declaration in module.type_declarations.values() {
            hit!(self.ty(&declaration.ty));
        }
        for global in module.global_declarations.values() {
            hit!(self.ty(&global.ty));
            if let Some(init) = &global.initializer {
                hit!(self.global_init(init));
            }
        }
        for function in module.functions.values() {
            hit!(self.function(function));
        }
        None
    }

    fn function(&mut self, function: &Function<'a>) -> Found {
        for param in &function.signature.params {
            hit!(self.ty(&param.ty));
        }
        hit!(self.ty(&function.signature.return_type));
        for block in function.basic_blocks.values() {
            for inst in &block.instructions {
                hit!(self.instruction(inst));
            }
        }
        None
    }

    fn global_init(&mut self, init: &GlobalInit<'a>) -> Found {
        match init {
            GlobalInit::Value(value) => self.value(value),
            GlobalInit::List(elements) => elements.iter().find_map(|e| self.init_element(e)),
            GlobalInit::Symbol(_) => None,
        }
    }

    fn init_element(&mut self, element: &InitElement<'a>) -> Found {
        match element {
            InitElement::Literal(literal) => literal_wide(literal),
            InitElement::Symbol(_) => None,
            InitElement::List(elements) => elements.iter().find_map(|e| self.init_element(e)),
        }
    }

    fn ty(&mut self, ty: &Type<'a>) -> Found {
        match ty {
            Type::Primitive(p) => wide_primitive(*p),
            Type::Named(name) => {
                if !self.named_seen.insert(name) {
                    return None;
                }
                let declaration = self.module.type_declarations.get(name)?;
                self.ty(&declaration.ty)
            }
            Type::Array { element_type, .. } => self.ty(element_type),
            Type::Vector { element_type, .. } => wide_primitive(*element_type),
            Type::Struct { fields, .. } => fields.iter().find_map(|f| self.ty(&f.ty)),
            Type::Void | Type::VaList => None,
        }
    }

    fn value(&mut self, value: &Value<'a>) -> Found {
        match value {
            Value::Constant(literal) => literal_wide(literal),
            Value::Variable(_) | Value::Global(_) => None,
        }
    }

    fn prim(&self, ty: &PrimitiveType) -> Found {
        wide_primitive(*ty)
    }

    fn instruction(&mut self, inst: &Instruction<'a>) -> Found {
        match inst {
            Instruction::Binary { ty, lhs, rhs, .. }
            | Instruction::OverflowFlag { ty, lhs, rhs, .. }
            | Instruction::Cmp { ty, lhs, rhs, .. } => {
                hit!(self.prim(ty));
                hit!(self.value(lhs));
                self.value(rhs)
            }
            Instruction::Unary { ty, value, .. } => {
                hit!(self.prim(ty));
                self.value(value)
            }
            Instruction::ZeroExtend {
                source_type,
                target_type,
                value,
                ..
            }
            | Instruction::SignExtend {
                source_type,
                target_type,
                value,
                ..
            }
            | Instruction::Trunc {
                source_type,
                target_type,
                value,
                ..
            }
            | Instruction::Bitcast {
                source_type,
                target_type,
                value,
                ..
            }
            | Instruction::IntToFloat {
                source_type,
                target_type,
                value,
                ..
            }
            | Instruction::FloatToInt {
                source_type,
                target_type,
                value,
                ..
            }
            | Instruction::FloatConvert {
                source_type,
                target_type,
                value,
                ..
            } => {
                hit!(self.prim(source_type));
                hit!(self.prim(target_type));
                self.value(value)
            }
            Instruction::Select {
                ty,
                cond,
                true_val,
                false_val,
                ..
            } => {
                hit!(self.ty(ty));
                hit!(self.value(cond));
                hit!(self.value(true_val));
                self.value(false_val)
            }
            Instruction::FuncAddr { .. } | Instruction::Unreachable => None,
            Instruction::CallIndirect {
                target,
                args,
                return_type,
                ..
            } => {
                hit!(self.value(target));
                for (ty, value) in args {
                    hit!(self.ty(ty));
                    hit!(self.value(value));
                }
                self.ty(return_type)
            }
            Instruction::Switch {
                ty, value, cases, ..
            } => {
                hit!(self.prim(ty));
                hit!(self.value(value));
                cases.iter().find_map(|(literal, _)| literal_wide(literal))
            }
            Instruction::Br { condition, .. } => self.value(condition),
            Instruction::Jmp { .. } => None,
            Instruction::Ret { ty, value } => {
                hit!(self.ty(ty));
                match value {
                    Some(value) => self.value(value),
                    None => None,
                }
            }
            Instruction::Alloc { allocated_ty, .. } => self.ty(allocated_ty),
            Instruction::Load { ty, ptr, .. } | Instruction::AtomicLoad { ty, ptr, .. } => {
                hit!(self.ty(ty));
                self.value(ptr)
            }
            Instruction::Store { ty, ptr, value, .. }
            | Instruction::AtomicStore { ty, ptr, value, .. } => {
                hit!(self.ty(ty));
                hit!(self.value(ptr));
                self.value(value)
            }
            Instruction::AtomicBinary { ty, ptr, value, .. } => {
                hit!(self.ty(ty));
                hit!(self.value(ptr));
                self.value(value)
            }
            Instruction::AtomicCompareExchange {
                ty,
                ptr,
                expected,
                desired,
                ..
            } => {
                hit!(self.ty(ty));
                hit!(self.value(ptr));
                hit!(self.value(expected));
                self.value(desired)
            }
            Instruction::Fence { .. } => None,
            Instruction::InlineAsm { operands, .. } => {
                for operand in operands {
                    if let AsmOperand::In { value, .. } = operand {
                        hit!(self.value(value));
                    }
                }
                None
            }
            Instruction::VectorBinary {
                element_type,
                lhs,
                rhs,
                ..
            } => {
                hit!(self.prim(element_type));
                hit!(self.value(lhs));
                self.value(rhs)
            }
            Instruction::VectorSplat {
                element_type,
                value,
                ..
            } => {
                hit!(self.prim(element_type));
                self.value(value)
            }
            Instruction::VectorExtractLane {
                element_type,
                vector,
                ..
            } => {
                hit!(self.prim(element_type));
                self.value(vector)
            }
            Instruction::VectorInsertLane {
                element_type,
                vector,
                scalar,
                ..
            } => {
                hit!(self.prim(element_type));
                hit!(self.value(vector));
                self.value(scalar)
            }
            Instruction::VectorShuffle {
                element_type,
                lhs,
                rhs,
                ..
            } => {
                hit!(self.prim(element_type));
                hit!(self.value(lhs));
                self.value(rhs)
            }
            Instruction::GetFieldPtr {
                struct_ptr,
                struct_type,
                ..
            } => {
                hit!(self.value(struct_ptr));
                self.ty(struct_type)
            }
            Instruction::GetElemPtr {
                array_ptr,
                index,
                element_type,
                ..
            } => {
                hit!(self.value(array_ptr));
                hit!(self.value(index));
                self.ty(element_type)
            }
            Instruction::PtrToInt {
                ptr_value,
                target_type,
                ..
            } => {
                hit!(self.value(ptr_value));
                self.prim(target_type)
            }
            Instruction::IntToPtr {
                int_value,
                target_type,
                ..
            } => {
                hit!(self.value(int_value));
                self.prim(target_type)
            }
            Instruction::Call { args, .. } => {
                for (ty, value) in args {
                    if let Some(ty) = ty {
                        hit!(self.ty(ty));
                    }
                    hit!(self.value(value));
                }
                None
            }
            Instruction::Copy { ty, value, .. } => {
                hit!(self.ty(ty));
                self.value(value)
            }
            Instruction::Write { buffer, size, .. } | Instruction::Read { buffer, size, .. } => {
                hit!(self.value(buffer));
                self.value(size)
            }
            Instruction::WriteByte { value, .. } => self.value(value),
            Instruction::ReadByte { .. } => None,
            Instruction::VaStart { ap } | Instruction::VaEnd { ap } => self.value(ap),
            Instruction::VaArg { ty, ap, .. } => {
                hit!(self.prim(ty));
                self.value(ap)
            }
            Instruction::VaCopy { dst, src } => {
                hit!(self.value(dst));
                self.value(src)
            }
        }
    }
}

fn literal_wide(literal: &Literal<'_>) -> Found {
    match literal {
        Literal::I128(_) => Some("i128"),
        Literal::U128(_) => Some("u128"),
        _ => None,
    }
}
