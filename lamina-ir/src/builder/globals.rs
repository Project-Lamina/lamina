use crate::{
    builder::{BuildErrorKind, IRBuilder, require},
    module::{GlobalDeclaration, GlobalInit, GlobalVisibility},
    types::{Literal, PrimitiveType, Type, Value},
};

impl<'a> IRBuilder<'a> {
    /// Declares a module-level global variable.
    ///
    /// # Panics
    /// Panics on a duplicate name or a malformed declaration.
    pub fn declare_global(&mut self, declaration: GlobalDeclaration<'a>) -> &mut Self {
        self.step(|builder| {
            let name = declaration.name;
            require(
                !builder.module.global_declarations.contains_key(name),
                BuildErrorKind::DuplicateGlobal,
                || format!("global '${name}' is already declared"),
            )?;
            if let Some(local) = declaration
                .visibility
                .filter(|v| matches!(v, GlobalVisibility::Internal | GlobalVisibility::Private))
            {
                require(!declaration.exported, BuildErrorKind::InvalidGlobal, || {
                    format!(
                        "Conflicting annotations on global ${name}: @export and {local} cannot be used together"
                    )
                })?;
                require(!declaration.weak, BuildErrorKind::InvalidGlobal, || {
                    format!(
                        "Conflicting annotations on global ${name}: @weak and {local} cannot be used together"
                    )
                })?;
            }
            if let Some(align) = declaration.align {
                require(align.is_power_of_two(), BuildErrorKind::InvalidGlobal, || {
                    format!(
                        "global '${name}' has @align({align}); alignment must be a power of two"
                    )
                })?;
            }
            if let Some(GlobalInit::Value(Value::Constant(Literal::String(text)))) =
                &declaration.initializer
            {
                require(
                    matches!(
                        &declaration.ty,
                        Type::Array { element_type, .. }
                            if matches!(
                                element_type.as_ref(),
                                Type::Primitive(PrimitiveType::I8 | PrimitiveType::Bool)
                            )
                    ),
                    BuildErrorKind::InvalidGlobal,
                    || {
                        format!(
                            "global '${name}' takes a string literal only in an [N x i8] or [N x bool] array"
                        )
                    },
                )?;
                require(!text.contains(['"', '\\']), BuildErrorKind::InvalidGlobal, || {
                    format!(
                        "string literal of global '${name}' contains '\"' or '\\'; the text IR reads them as the end of the literal or an escape"
                    )
                })?;
            }
            builder.module.global_declarations.insert(name, declaration);
            Ok(())
        })
    }

    /// Declares a NUL-terminated byte array holding `text`.
    ///
    /// # Panics
    /// Panics when the name is already declared.
    pub fn declare_string_global(&mut self, name: &'a str, text: &'a str) -> &mut Self {
        self.declare_global(GlobalDeclaration {
            name,
            ty: Type::Array {
                element_type: Box::new(Type::Primitive(PrimitiveType::I8)),
                size: text.len() as u64 + 1,
            },
            is_extern: false,
            initializer: Some(GlobalInit::Value(Value::Constant(Literal::String(text)))),
            exported: false,
            visibility: None,
            weak: false,
            align: None,
            thread_local: None,
        })
    }
}
