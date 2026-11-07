//! x86-64 microarchitecture levels: the instruction-set baseline a generated binary may rely on.

use std::{fmt, str::FromStr};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum X86CpuLevel {
    V1,
    #[default]
    V2,
    V3,
    V4,
}

impl X86CpuLevel {
    pub const ALL: [X86CpuLevel; 4] = [Self::V1, Self::V2, Self::V3, Self::V4];

    pub fn features(self) -> &'static str {
        match self {
            Self::V1 => "SSE2",
            Self::V2 => "SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT, CMPXCHG16B",
            Self::V3 => "v2 + AVX, AVX2, BMI1, BMI2, F16C, FMA, LZCNT, MOVBE",
            Self::V4 => "v3 + AVX-512F, AVX-512BW, AVX-512CD, AVX-512DQ, AVX-512VL",
        }
    }
}

impl fmt::Display for X86CpuLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = match self {
            Self::V1 => 1,
            Self::V2 => 2,
            Self::V3 => 3,
            Self::V4 => 4,
        };
        write!(f, "x86-64-v{n}")
    }
}

impl FromStr for X86CpuLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_ascii_lowercase();
        let digit = lower
            .strip_prefix("x86-64-v")
            .or_else(|| lower.strip_prefix('v'))
            .unwrap_or("");
        match digit {
            "1" => Ok(Self::V1),
            "2" => Ok(Self::V2),
            "3" => Ok(Self::V3),
            "4" => Ok(Self::V4),
            _ => Err(format!(
                "unknown CPU level '{s}' (expected v1, v2, v3 or v4)"
            )),
        }
    }
}

/// What an x86-64 target has: the psABI microarchitecture level, plus the extensions
/// that belong to no level and so have to be named on their own.
///
/// AVX-512-FP16 and AVX-512-BF16 are not on the v1..v4 ladder. FP16 arrives with
/// Sapphire Rapids; BF16 with Cooper Lake, Sapphire Rapids and Zen 4. A part may have
/// BF16 without FP16, so neither implies the other, and both need v4 underneath them
/// because they are AVX-512 encodings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct X86Features {
    pub level: X86CpuLevel,
    /// AVX-512-FP16: half-precision arithmetic, not just conversion.
    pub avx512_fp16: bool,
    /// AVX-512-BF16: bfloat16 conversion and dot product.
    pub avx512_bf16: bool,
}

impl X86Features {
    pub const fn at(level: X86CpuLevel) -> Self {
        Self {
            level,
            avx512_fp16: false,
            avx512_bf16: false,
        }
    }

    /// Whether the level underneath these extensions can encode them. Both are
    /// AVX-512, so neither means anything below v4.
    pub const fn extensions_are_encodable(self) -> bool {
        !(self.avx512_fp16 || self.avx512_bf16) || matches!(self.level, X86CpuLevel::V4)
    }
}

impl From<X86CpuLevel> for X86Features {
    fn from(level: X86CpuLevel) -> Self {
        Self::at(level)
    }
}

impl fmt::Display for X86Features {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.level)?;
        if self.avx512_fp16 {
            write!(f, "+fp16")?;
        }
        if self.avx512_bf16 {
            write!(f, "+bf16")?;
        }
        Ok(())
    }
}

impl FromStr for X86Features {
    type Err = String;

    /// `v3`, `x86-64-v4`, `v4+bf16`, `v4+fp16+bf16`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut parts = s.split('+');
        let level: X86CpuLevel = parts.next().unwrap_or_default().parse()?;
        let mut out = Self::at(level);
        for extension in parts {
            match extension.to_ascii_lowercase().as_str() {
                "fp16" | "avx512fp16" => out.avx512_fp16 = true,
                "bf16" | "avx512bf16" => out.avx512_bf16 = true,
                other => {
                    return Err(format!(
                        "unknown x86 extension '{other}' (expected fp16 or bf16)"
                    ));
                }
            }
        }
        if !out.extensions_are_encodable() {
            return Err(format!(
                "{} needs x86-64-v4 underneath it, which is where AVX-512 starts",
                if out.avx512_fp16 { "fp16" } else { "bf16" }
            ));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_short_and_psabi_names() {
        assert_eq!("v1".parse(), Ok(X86CpuLevel::V1));
        assert_eq!("V2".parse(), Ok(X86CpuLevel::V2));
        assert_eq!("x86-64-v3".parse(), Ok(X86CpuLevel::V3));
        assert_eq!("X86-64-V4".parse(), Ok(X86CpuLevel::V4));
        assert!("v5".parse::<X86CpuLevel>().is_err());
        assert!("native".parse::<X86CpuLevel>().is_err());
    }

    #[test]
    fn levels_are_ordered_and_default_to_v2() {
        assert!(X86CpuLevel::V1 < X86CpuLevel::V2);
        assert!(X86CpuLevel::V4 > X86CpuLevel::V3);
        assert_eq!(X86CpuLevel::default(), X86CpuLevel::V2);
        assert_eq!(X86CpuLevel::V2.to_string(), "x86-64-v2");
    }

    /// AVX-512-FP16 and AVX-512-BF16 are not on the v1..v4 ladder, so a target names
    /// them beside the level rather than reaching them by going up.
    #[test]
    fn x86_extensions_are_named_beside_the_level() {
        assert_eq!(X86Features::at(X86CpuLevel::V4), "v4".parse().unwrap());
        let bf16: X86Features = "v4+bf16".parse().unwrap();
        assert!(bf16.avx512_bf16 && !bf16.avx512_fp16);
        assert_eq!(bf16.level, X86CpuLevel::V4);
        let both: X86Features = "x86-64-v4+fp16+bf16".parse().unwrap();
        assert!(both.avx512_fp16 && both.avx512_bf16);
        assert_eq!(both.to_string(), "x86-64-v4+fp16+bf16");
        assert!("v4+sve".parse::<X86Features>().is_err());
    }

    /// Both are AVX-512 encodings, so neither means anything below v4.
    #[test]
    fn an_avx512_extension_below_v4_is_refused() {
        for level in ["v1", "v2", "v3"] {
            let error = format!("{level}+bf16").parse::<X86Features>().unwrap_err();
            assert!(error.contains("x86-64-v4"), "{error}");
        }
        assert!("v4+bf16".parse::<X86Features>().is_ok());
    }
}
