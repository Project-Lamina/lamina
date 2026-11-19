//! Wall-clock totals per compiler phase, printed when `LAMINA_TIME_PASSES` is set.

use std::{
    collections::BTreeMap,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Default)]
struct Total {
    elapsed: Duration,
    calls: u64,
}

static TOTALS: Mutex<BTreeMap<&'static str, Total>> = Mutex::new(BTreeMap::new());

pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("LAMINA_TIME_PASSES")
            .is_some_and(|value| !value.is_empty() && value != "0")
    })
}

/// Runs `work` and adds its wall time to the total for `phase`. Calls from several
/// threads add up, so a phase run in parallel can total more than the elapsed time.
pub fn measure<T>(phase: &'static str, work: impl FnOnce() -> T) -> T {
    if !enabled() {
        return work();
    }
    let start = Instant::now();
    let result = work();
    let elapsed = start.elapsed();
    if let Ok(mut totals) = TOTALS.lock() {
        let total = totals.entry(phase).or_default();
        total.elapsed += elapsed;
        total.calls += 1;
    }
    result
}

/// Prints the accumulated totals to stderr. Does nothing when timing is off.
pub fn report() {
    if !enabled() {
        return;
    }
    let Ok(totals) = TOTALS.lock() else {
        return;
    };
    eprintln!("[TIME] {:<34} {:>12} {:>10}", "phase", "ms", "calls");
    for (phase, total) in totals.iter() {
        eprintln!(
            "[TIME] {:<34} {:>12.1} {:>10}",
            phase,
            total.elapsed.as_secs_f64() * 1000.0,
            total.calls
        );
    }
}
