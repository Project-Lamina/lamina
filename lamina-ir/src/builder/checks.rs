//! Shape rules shared by the IR builders. Each check fails with the rule the text parser enforces.

use crate::{
    builder::{BuildError, BuildErrorKind, require},
    types::PrimitiveType,
};

pub(crate) fn is_signed(ty: PrimitiveType) -> bool {
    matches!(
        ty,
        PrimitiveType::I8
            | PrimitiveType::I16
            | PrimitiveType::I32
            | PrimitiveType::I64
            | PrimitiveType::I128
    )
}

pub(crate) fn integer_bits(ty: PrimitiveType) -> Option<u32> {
    match ty {
        PrimitiveType::Bool => Some(1),
        PrimitiveType::I8 | PrimitiveType::U8 => Some(8),
        PrimitiveType::I16 | PrimitiveType::U16 => Some(16),
        PrimitiveType::I32 | PrimitiveType::U32 => Some(32),
        PrimitiveType::I64 | PrimitiveType::U64 => Some(64),
        PrimitiveType::I128 | PrimitiveType::U128 => Some(128),
        _ => None,
    }
}

pub(crate) fn scalar_bits(ty: PrimitiveType) -> Option<u32> {
    match ty {
        PrimitiveType::F32 => Some(32),
        PrimitiveType::F64 => Some(64),
        PrimitiveType::Bool => None,
        _ => integer_bits(ty),
    }
}

pub(crate) fn integer_signedness(ty: PrimitiveType) -> Option<bool> {
    match ty {
        PrimitiveType::I8
        | PrimitiveType::I16
        | PrimitiveType::I32
        | PrimitiveType::I64
        | PrimitiveType::I128 => Some(true),
        PrimitiveType::U8
        | PrimitiveType::U16
        | PrimitiveType::U32
        | PrimitiveType::U64
        | PrimitiveType::U128 => Some(false),
        _ => None,
    }
}

/// Panics with the error when a check failed, for the owned builder, which has no fallible mode.
pub(crate) fn enforce(check: Result<(), BuildError>) {
    if let Err(error) = check {
        panic!("{error}");
    }
}

pub(crate) fn check_zext(source: PrimitiveType, target: PrimitiveType) -> Result<(), BuildError> {
    require(!is_signed(source), BuildErrorKind::InvalidType, || {
        format!(
            "zext zero-extends an unsigned or bool source, found {source}; use sext for a signed source"
        )
    })?;
    check_extend(source, target)
}

pub(crate) fn check_sext(source: PrimitiveType, target: PrimitiveType) -> Result<(), BuildError> {
    require(is_signed(source), BuildErrorKind::InvalidType, || {
        format!(
            "sext sign-extends a signed source, found {source}; use zext for an unsigned or bool source"
        )
    })?;
    check_extend(source, target)
}

fn check_extend(source: PrimitiveType, target: PrimitiveType) -> Result<(), BuildError> {
    let source_bits = integer_bits(source).ok_or_else(|| {
        BuildError::new(
            BuildErrorKind::InvalidType,
            format!("extend source must be an integer or bool, found {source}"),
        )
    })?;
    let target_bits = match target {
        PrimitiveType::Bool => None,
        _ => integer_bits(target),
    }
    .ok_or_else(|| {
        BuildError::new(
            BuildErrorKind::InvalidType,
            format!("extend target must be an integer, found {target}"),
        )
    })?;
    require(
        target_bits > source_bits,
        BuildErrorKind::InvalidType,
        || format!("extend from {source} to {target} must widen; use trunc to narrow"),
    )
}

pub(crate) fn check_trunc(source: PrimitiveType, target: PrimitiveType) -> Result<(), BuildError> {
    let source_bits = match source {
        PrimitiveType::Bool => None,
        _ => integer_bits(source),
    }
    .ok_or_else(|| {
        BuildError::new(
            BuildErrorKind::InvalidType,
            format!("trunc source must be an integer, found {source}"),
        )
    })?;
    let target_bits = match target {
        PrimitiveType::Bool => None,
        _ => integer_bits(target),
    }
    .ok_or_else(|| {
        BuildError::new(
            BuildErrorKind::InvalidType,
            format!("trunc target must be an integer, found {target}"),
        )
    })?;
    require(
        target_bits < source_bits,
        BuildErrorKind::InvalidType,
        || format!("trunc from {source} to {target} must narrow; use extend to widen"),
    )
}

pub(crate) fn check_bitcast(
    source: PrimitiveType,
    target: PrimitiveType,
) -> Result<(), BuildError> {
    if let (Some(source_bits), Some(target_bits)) = (scalar_bits(source), scalar_bits(target)) {
        require(
            source_bits == target_bits,
            BuildErrorKind::InvalidType,
            || {
                format!(
                    "bitcast reinterprets equal widths, found {source} ({source_bits} bits) to {target} ({target_bits} bits)"
                )
            },
        )?;
    }
    Ok(())
}
