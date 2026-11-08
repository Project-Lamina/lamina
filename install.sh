#!/bin/sh
# Install the Lamina toolchain: the compiler and the assembler and linker it drives.
#
#   curl -fsSL https://lamina.sh/install.sh | sh
#   curl -fsSL https://lamina.sh/install.sh | sh -s -- --version 0.2.1
#   curl -fsSL https://lamina.sh/install.sh | sh -s -- --with-ldb
#
# `cargo install lamina` alone leaves whatever `ras` is already on PATH, and an older
# one makes the compiler fall back to GNU as and ld without being asked. This installs
# the three together and checks afterwards that they agree.

set -eu

CRATES="lamina lamina-ras lamina-weld"
VERSION=""
WITH_LDB=0
DRY_RUN=0

say() { printf '%s\n' "$*"; }
err() { printf 'install.sh: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'USAGE'
Usage: install.sh [options]

  --version <v>   Install this version of lamina instead of the newest. ras and weld
                  are always the newest, because lamina refuses one older than it
                  needs and never refuses a newer one
  --with-ldb      Also install ldb, the debugger. It is not published yet, so it
                  is built from its repository and --version does not apply
  --dry-run       Print the commands instead of running them
  -h, --help      Print this message

Installs lamina, ras and weld into the cargo bin directory, which is
$CARGO_HOME/bin, or ~/.cargo/bin when CARGO_HOME is unset.
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --version) [ $# -ge 2 ] || err "--version needs a version"; VERSION="$2"; shift 2 ;;
        --version=*) VERSION="${1#--version=}"; shift ;;
        --with-ldb) WITH_LDB=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) err "unknown option '$1' (try --help)" ;;
    esac
done

command -v cargo >/dev/null 2>&1 || err \
"cargo is not on PATH. Lamina installs from source, so it needs a Rust toolchain:
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"

bin_dir="${CARGO_HOME:-$HOME/.cargo}/bin"

run() {
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '  %s\n' "$*"
    else
        "$@"
    fi
}

say "Installing the Lamina toolchain into $bin_dir"
for crate in $CRATES; do
    if [ -n "$VERSION" ] && [ "$crate" = lamina ]; then
        run cargo install "$crate" --version "$VERSION" --force
    else
        run cargo install "$crate" --force
    fi
done

# ldb is not on crates.io yet, so it comes from its repository and takes no --version.
if [ "$WITH_LDB" -eq 1 ]; then
    run cargo install --git https://github.com/Project-Lamina/ldb --force
fi

# `lamina` ships a `weld` of its own, so whichever crate was installed last wins. Report
# what is actually on PATH rather than what was asked for. This half only reads, so a
# dry run does it too and says what is there now.
say ""
if [ "$DRY_RUN" -eq 1 ]; then
    say "Already on PATH:"
else
    say "Installed:"
fi
mismatch=0
for tool in lamina ras weld ldb; do
    path=$(command -v "$tool" 2>/dev/null || true)
    [ -n "$path" ] || continue
    case "$path" in
        "$bin_dir"/*) where="" ;;
        *) where=" (from $path, not $bin_dir)"; mismatch=1 ;;
    esac
    say "  $("$tool" --version 2>/dev/null || printf '%s ?' "$tool")$where"
done

case ":$PATH:" in
    *":$bin_dir:"*) ;;
    *)
        say ""
        say "$bin_dir is not on PATH. Add it:"
        say "  export PATH=\"$bin_dir:\$PATH\""
        mismatch=1
        ;;
esac

if [ "$mismatch" -eq 1 ]; then
    say ""
    say "A tool found outside $bin_dir shadows the one there, and lamina will reach that"
    say "one. Fix PATH, or remove the other copy, before reporting a version warning."
fi
