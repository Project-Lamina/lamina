//! Global variable declarations for IR builder.
//!
//! Globals belong to the module rather than to the function being built. Each declaration
//! is checked against the rules the text parser applies to a global.

use crate::{
    builder::IRBuilder,
    module::{GlobalDeclaration, GlobalInit, GlobalVisibility},
    types::{Literal, PrimitiveType, Type, Value},
};

impl<'a> IRBuilder<'a> {
    /// Declares a module-level global variable (`global $name: ty = init`).
    ///
    /// Panics when the name is already declared, when `@export` or `@weak` is combined with
    /// `@internal` or `@private`, when `align` is not a power of two, or when a string
    /// initializer does not fit a byte array of `i8` or `bool`.
    pub fn declare_global(&mut self, declaration: GlobalDeclaration<'a>) -> &mut Self {
        let name = declaration.name;
        assert!(
            !self.module.global_declarations.contains_key(name),
            "global '${name}' is already declared"
        );
        if let Some(local) = declaration
            .visibility
            .filter(|v| matches!(v, GlobalVisibility::Internal | GlobalVisibility::Private))
        {
            assert!(
                !declaration.exported,
                "Conflicting annotations on global ${name}: @export and {local} cannot be used together"
            );
            assert!(
                !declaration.weak,
                "Conflicting annotations on global ${name}: @weak and {local} cannot be used together"
            );
        }
        if let Some(align) = declaration.align {
            assert!(
                align.is_power_of_two(),
                "global '${name}' has @align({align}); alignment must be a power of two"
            );
        }
        if let Some(GlobalInit::Value(Value::Constant(Literal::String(text)))) =
            &declaration.initializer
        {
            assert!(
                matches!(
                    &declaration.ty,
                    Type::Array { element_type, .. }
                        if matches!(
                            element_type.as_ref(),
                            Type::Primitive(PrimitiveType::I8 | PrimitiveType::Bool)
                        )
                ),
                "global '${name}' takes a string literal only in an [N x i8] or [N x bool] array"
            );
            assert!(
                !text.contains(['"', '\\']),
                "string literal of global '${name}' contains '\"' or '\\'; the text IR reads them as the end of the literal or an escape"
            );
        }
        self.module.global_declarations.insert(name, declaration);
        self
    }

    /// Declares a byte array holding `text` and its NUL terminator (`global $name: [N x i8] = "text"`).
    ///
    /// `N` is the length of `text` in bytes plus one, the terminator the text IR leaves implicit.
    /// Panics under the same conditions as `declare_global`.
    pub fn declare_string_global(&mut self, name: &'a str, text: &'a str) -> &mut Self {
        self.declare_global(GlobalDeclaration {
            name,
            ty: Type::Array {
                element_type: Box::new(Type::Primitive(PrimitiveType::I8)),
                size: text.len() as u64 + 1,
            },
            initializer: Some(GlobalInit::Value(Value::Constant(Literal::String(text)))),
            exported: false,
            visibility: None,
            weak: false,
            align: None,
            thread_local: None,
        })
    }
}
