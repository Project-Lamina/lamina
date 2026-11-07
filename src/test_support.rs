#![allow(clippy::expect_used)]

#[path = "test_support_macho.rs"]
pub mod macho;

use std::{
    env, fs,
    path::{Path, PathBuf},
    process,
    sync::atomic::{AtomicUsize, Ordering},
    thread,
};

static NEXT: AtomicUsize = AtomicUsize::new(0);

pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    pub fn new(label: &str) -> Self {
        let base = option_env!("CARGO_TARGET_TMPDIR")
            .unwrap_or(concat!(env!("CARGO_MANIFEST_DIR"), "/target/test-tmp"));
        fs::create_dir_all(base).expect("create test temp base");
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = Path::new(base).join(format!("{label}-{}-{n}", process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create test temp dir");
        Self { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        // Keep the artifacts of a failing test for debugging.
        if thread::panicking() {
            return;
        }
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// The `lamina` executable beside this test binary. `cargo test` builds it as part of
/// the bin target; `cargo test --lib` does not, so say which command is missing rather
/// than failing later on an absent program.
pub fn lamina_binary() -> PathBuf {
    let mut path = env::current_exe().expect("test binary path");
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    let lamina = path.join("lamina");
    assert!(
        lamina.is_file(),
        "{} is not built: run `cargo build --bin lamina`",
        lamina.display()
    );
    lamina
}
