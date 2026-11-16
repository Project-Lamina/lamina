#!/usr/bin/env python3
"""
Lamina Test Runner
Runs test cases and verifies their expected outputs.
"""

import os
import platform
import shutil
import subprocess
import sys
import glob
import argparse
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode the emoji below.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

# Testcases that cannot run on s390x, with the reason. Any other testcase must compile,
# run and match its expected output, or have a s390x variant (see CROSS_TARGETS).
# These pin a platform's C ABI for aggregates holding float fields. A backend
# that declares no float-aggregate ABI refuses them by name, which is correct,
# so running them there reports a failure that can never be fixed by fixing the
# backend under test.
FLOAT_AGGREGATE_UNSUPPORTED = {
    name: 'the backend does not place aggregates with float fields'
    for name in (
        'aggregate_float_params',
        'aggregate_homogeneous_params',
        'aggregate_homogeneous_spill',
        'win64_float_aggregate_args',
        'win64_float_aggregate_returns',
    )
}

# A target with 32-bit integer registers refuses, by name, every testcase whose
# shape needs a wider register than it has or a facility its backend does not
# implement. These are backend gaps, not wrong answers: the compile fails rather
# than producing a number, so running them reports a failure no fix to the code
# under test can clear.
ILP32_UNSUPPORTED = {
    **{
        name: 'an i64 atomic needs a double-word atomic the backend does not emit'
        for name in ('atomic_cmpxchg', 'atomic_orderings', 'atomics')
    },
    'atomic_live_result': 'this backend does not emit atomics',
    'vector_operations': 'this backend has no vector unit',
}

I686_UNSUPPORTED = {
    **ILP32_UNSUPPORTED,
}

ARMV7_UNSUPPORTED = {
    'inline_asm_rdtsc': 'rdtsc reads the x86 timestamp counter; armv7 reads the cycle counter only from a privileged mode',
    **FLOAT_AGGREGATE_UNSUPPORTED,
    **ILP32_UNSUPPORTED,
}

S390X_UNSUPPORTED = {
    'vector_operations': 'the z vector facility (z13 and later) is not implemented',
}


# Cross targets run through `--target <name>`: each program is compiled for the
# target, then executed under its user-mode emulator. Expected files are chosen
# by the target's byte order, not the host's. A testcase `<name>.<target>.lamina`
# is compiled in place of `<name>.lamina` for that target and shares its expected
# output; variant files are never run as testcases of their own.
RISCV64_UNSUPPORTED = {
    'vector_operations': 'vectors need the V extension, which rv64imafd does not carry',
}

AARCH64_UNSUPPORTED: dict[str, str] = {}


CROSS_TARGETS = {
    's390x_linux': {
        'byteorder': 'big',
        'compiler': 's390x-linux-gnu-gcc',
        'emulator': 'qemu-s390x-static',
        'sysroot': '/usr/s390x-linux-gnu',
        'unsupported': S390X_UNSUPPORTED,
    },
    'aarch64_linux': {
        'byteorder': 'little',
        'compiler': 'aarch64-linux-gnu-gcc',
        'emulator': 'qemu-aarch64-static',
        'sysroot': '/usr/aarch64-linux-gnu',
        'unsupported': AARCH64_UNSUPPORTED,
    },
    'riscv64_linux': {
        'byteorder': 'little',
        'compiler': 'riscv64-linux-gnu-gcc',
        'emulator': 'qemu-riscv64-static',
        'sysroot': '/usr/riscv64-linux-gnu',
        'unsupported': RISCV64_UNSUPPORTED,
    },
    'armv7_linux': {
        'byteorder': 'little',
        'compiler': 'arm-linux-gnueabihf-gcc',
        'emulator': 'qemu-arm-static',
        'sysroot': '/usr/arm-linux-gnueabihf',
        'unsupported': ARMV7_UNSUPPORTED,
    },
    # i686 runs natively on an x86_64 Linux host, so there is no emulator and no
    # sysroot: the 32-bit crt and libraries come from the host's own multilib set.
    'i686_linux': {
        'byteorder': 'little',
        'compiler': 'gcc',
        'emulator': None,
        'sysroot': None,
        'crt1': '/usr/lib32/crt1.o',
        'hosts': {('Linux', 'x86_64'), ('Linux', 'i686')},
        'unsupported': I686_UNSUPPORTED,
    },
    # Lamina links PE output itself with weld, so `compiler` names that linker.
    # wine runs the executable directly, with no sysroot to point at.
    'x86_64_windows': {
        'byteorder': 'little',
        'compiler': 'weld',
        'emulator': 'wine',
        'sysroot': None,
        'unsupported': {},
    },
}

# A native run compiles for the host with no --target, so it takes the variant for
# the host's own target. Keyed by (platform.system(), platform.machine()).
HOST_VARIANTS = {
    ('Darwin', 'arm64'): 'aarch64_macos',
    ('Linux', 'aarch64'): 'aarch64_linux',
    ('Windows', 'AMD64'): 'x86_64_windows',
}

VARIANT_TARGETS = set(CROSS_TARGETS) | set(HOST_VARIANTS.values())

def host_variant():
    """The variant a native run on this machine selects, or None."""
    return HOST_VARIANTS.get((platform.system(), platform.machine()))

def variant_target(test_path):
    """The target that `<name>.<target>.lamina` is a variant for, or None for a base testcase."""
    stem = Path(test_path).stem
    _, dot, suffix = stem.rpartition('.')
    return suffix if dot and suffix in VARIANT_TARGETS else None

def testcase_source(test_path, target=None):
    """The file to compile for `target`: its variant when one exists, else the testcase."""
    test_path = Path(test_path)
    if target:
        variant = test_path.with_name(f"{test_path.stem}.{target}.lamina")
        if variant.exists():
            return str(variant)
    return str(test_path)

def discover(pattern):
    """Base testcases matching `pattern`; variant files are not testcases of their own."""
    return [p for p in sorted(glob.glob(pattern)) if variant_target(p) is None]

def cross_toolchain_problem(target):
    """Return why `target` cannot be built and run here, or None when it can."""
    spec = CROSS_TARGETS[target]
    hosts = spec.get('hosts')
    if hosts and (platform.system(), platform.machine()) not in hosts:
        return f"{target} runs only on " + " or ".join(sorted(f"{s} {m}" for s, m in hosts))
    missing = []
    for tool in (spec['compiler'], emulator_command(spec)):
        if tool is not None and shutil.which(tool) is None:
            missing.append(tool)
    crt1 = spec.get('crt1')
    if crt1 is None and spec['sysroot']:
        crt1 = str(Path(spec['sysroot']) / 'lib' / 'crt1.o')
    if crt1 and not Path(crt1).is_file():
        missing.append(crt1)
    if missing:
        return "missing " + ", ".join(missing)
    return None


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_colored(message, color):
    print(f"{color}{message}{Colors.END}")

def run_command(cmd, cwd=None):
    """Run a command and return (success, stdout, stderr)"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            # Tests that read stdin without declared input must see EOF rather
            # than inherit a terminal and block until the timeout.
            stdin=subprocess.DEVNULL,
            timeout=60,
            errors='replace'
        )
        return (
            result.returncode == 0,
            result.stdout.strip(),
            result.stderr.strip(),
            result.returncode,
        )
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out", None
    except Exception as e:
        return False, "", str(e), None

def describe_exit(code):
    """Human-readable exit status. Windows reports crashes as large NTSTATUS values."""
    if code is None:
        return "no exit status"
    if code < 0:
        return f"killed by signal {-code}"
    known = {
        0xC0000005: "access violation",
        0xC000001D: "illegal instruction",
        0xC00000FD: "stack overflow",
        0xC0000094: "integer divide by zero",
        0xC0000374: "heap corruption",
    }
    unsigned = code & 0xFFFFFFFF
    name = known.get(unsigned)
    detail = f" ({name})" if name else ""
    return f"exit code {code} [0x{unsigned:08X}]{detail}"

def is_crash(code):
    """True when the process died rather than chose its exit status.

    Testcases may return any value from main (stdin.lamina returns the first
    byte it read), so a plain non-zero exit is not a failure on its own.
    """
    if code is None:
        return True
    if code < 0:
        return True
    return (code & 0xFFFFFFFF) >= 0xC0000000

def record_crash(executable_path, code):
    """Append a crashed binary to target/crashes.txt so CI can debug it.

    A crash leaves no stdout and often no stderr, so the exit status alone says
    nothing about where it died. The workflow re-runs these under a debugger.
    """
    try:
        path = Path(__file__).parent / 'target' / 'crashes.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(f"{executable_path}\t{describe_exit(code)}\n")
    except OSError:
        pass

def execution_failure(stdout, stderr, code):
    parts = [describe_exit(code)]
    if stderr:
        parts.append(f"stderr: {stderr}")
    if stdout:
        parts.append(f"stdout: {stdout}")
    return "Execution failed: " + "; ".join(parts)

def load_expected_output(test_path, byteorder=sys.byteorder, target=None):
    """Load expected output from .expected or expected_output.txt file"""
    test_path = Path(test_path)

    # A <stem>.<target>.expected answers for a target whose C ABI gives a program a
    # different answer, such as the 4-byte alignment i386 gives a 64-bit field. A
    # <stem>.<byteorder>.expected answers for programs of that byte order.
    expected_path = test_path.with_suffix('.expected')
    for name in (target, byteorder):
        variant = test_path.with_name(f"{test_path.stem}.{name}.expected")
        if name and variant.exists():
            expected_path = variant
            break
    
    # If not found, try expected_output.txt (standard for benchmarks)
    if not expected_path.exists():
        expected_path = test_path.parent / "expected_output.txt"
        
    if not expected_path.exists():
        return None

    if test_path.name == 'io_types.lamina':
        with open(expected_path, 'rb') as f:
            return f.read().replace(b"\r\n", b"\n")

    with open(expected_path, 'rb') as f:
        text = f.read().decode('utf-8', errors='replace')

    expected_values = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            expected_values.append(line)

    return expected_values

def emulator_command(spec):
    """The user-mode emulator to run a cross binary with, or None.

    Debian packaged the `qemu-<arch>-static` binaries separately for years; on
    Ubuntu 26 `qemu-user-static` became a virtual package and the dynamic
    `qemu-<arch>` is what gets installed. Either runs our binaries, so take
    whichever is present.
    """
    name = spec['emulator']
    if name is None or shutil.which(name):
        return name
    plain = name.removesuffix('-static')
    return plain if shutil.which(plain) else name


def execution_argv(executable_path, target=None):
    """Command line that runs a built program on the host or under a cross emulator."""
    if target is None:
        return [executable_path]
    spec = CROSS_TARGETS[target]
    emulator = emulator_command(spec)
    if emulator is None:
        return [executable_path]
    if spec['sysroot'] is None:
        return [emulator, executable_path]
    return [emulator, '-L', spec['sysroot'], executable_path]

def run_timeout(target):
    """Seconds a built program gets. Benchmarks are sized so the fastest native
    compiler lands near 200ms on x86_64; an emulator costs one to two orders, and
    a 32-bit target splits every i64 operation into a register pair."""
    return 60 if target is None else 900

# Stdin for the testcases that read it, so each run sees the same input.
STDIN_INPUTS = {
    'stdin.lamina': 'A\nBUFFER_TEST\nB',
    'io_buffer.lamina': 'Buf',
    'readbyte_nul.lamina': 'a\x00b',
}

def run_program(argv, cwd, stdin_input=None, timeout=60, env=None):
    """Run a program and return (success, stdout, stderr, exit_code)."""
    stdin_arg = {'input': stdin_input} if stdin_input is not None else {'stdin': subprocess.DEVNULL}
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
            errors='replace',
            env=env,
            **stdin_arg,
        )
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out", None
    except Exception as e:
        return False, "", str(e), None
    return (
        result.returncode == 0,
        result.stdout.strip(),
        result.stderr.strip(),
        result.returncode,
    )

def run_raw_output(argv, project_root, env=None):
    """Run a binary whose stdout is compared byte for byte (write sends raw memory)."""
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            cwd=project_root,
            stdin=subprocess.DEVNULL,
            timeout=60,
            env=env,
        )
    except Exception as e:
        return False, f"Execution failed: {str(e)}"
    if is_crash(result.returncode):
        return False, execution_failure(
            result.stdout.decode('utf-8', errors='replace'),
            result.stderr.decode('utf-8', errors='replace'),
            result.returncode,
        )
    return True, result.stdout.replace(b"\r\n", b"\n")

# Testcases `lamina --jit` runs in-process; each must print what its AOT build prints.
# The JIT mode is x86_64 Linux only for now, and runs only these testcases.
JIT_TESTCASES = {
    'aggregate_load_store.lamina', 'arithmetic.lamina',
    'binary_op_rhs_in_destination.lamina', 'call_typed_arguments.lamina',
    'calling_convention_bug_test.lamina', 'capable_lamina.lamina',
    'checked_arithmetic.lamina', 'complex_arithmetic.lamina',
    'conditionals.lamina', 'constants.lamina',
    'dealloc_stack_vs_heap_bug_test.lamina', 'debug_printing.lamina',
    'div_by_constant.lamina', 'division_constant_divisors.lamina',
    'division_runtime_narrowing.lamina', 'division_small_types_bug_test.lamina',
    'edge_cases.lamina', 'edge_cases_bug_test.lamina',
    'edge_cases_extra.lamina', 'empty_struct_type.lamina',
    'extern_declarations.lamina', 'external_functions.lamina',
    'f64_literal_width.lamina', 'float_exponent_literals.lamina',
    'float_nan_comparisons.lamina', 'float_phi_copy.lamina',
    'float_width_conversions.lamina', 'floating_point_bug_test.lamina',
    'floating_point_register_bug_test.lamina', 'fptoint_signed.lamina',
    'function_annotations.lamina', 'function_call_isolation.lamina',
    'function_placement_annotations.lamina', 'functions.lamina',
    'gep_element_size_bug_test.lamina',
    'gep_element_size_comprehensive_bug_test.lamina',
    'gep_variable_indices.lamina', 'getfieldptr_offset_bug_test.lamina',
    'getfieldptr_param_layout.lamina', 'global_element_list.lamina',
    'global_literal_fills_slot.lamina', 'global_nested_init.lamina',
    'global_pointer_addend.lamina', 'global_pointer_fields.lamina',
    'global_pointer_relocation.lamina', 'global_readonly_string.lamina',
    'global_string.lamina', 'global_struct_init.lamina',
    'globals_address.lamina', 'globals_export.lamina', 'globals_float.lamina',
    'globals_int.lamina', 'globals_zero_init.lamina',
    'heap_allocation_bug_test.lamina', 'heap_memory.lamina',
    'heapfree_reassigned_heap.lamina', 'heapfree_reassigned_stack.lamina',
    'hello_world.lamina', 'i8_signedness.lamina', 'inttofp_signed.lamina',
    'io_basic.lamina', 'io_buffer.lamina', 'io_comprehensive.lamina',
    'io_error_handling.lamina', 'io_pointer.lamina', 'io_types.lamina',
    'io_write.lamina', 'lamina_demo.lamina', 'large_basic_block.lamina',
    'large_constants.lamina', 'large_stack_frame.lamina',
    'leaf_argument_registers.lamina', 'load_store_alignment.lamina',
    'load_store_operations.lamina', 'load_store_optimization_bug_test.lamina',
    'loop_rotation_bound_change.lamina', 'loop_rotation_counted.lamina',
    'loop_rotation_header_call.lamina', 'loop_rotation_memory_test.lamina',
    'loop_rotation_multi_latch.lamina', 'loop_rotation_nested.lamina',
    'loops.lamina', 'mem2reg.lamina', 'memory_intrinsics.lamina',
    'memory_test.lamina', 'narrow_add_renormalizes.lamina',
    'narrow_store_width.lamina', 'narrow_unsigned_canonical.lamina',
    'negative_multiplier.lamina', 'neon_vectorized_loops.lamina',
    'nested_calls.lamina', 'odd_sized_global_addresses.lamina',
    'output_freestanding.lamina', 'output_via_libc.lamina',
    'pointer_arithmetic_brainfuck.lamina',
    'pointer_from_literal_address.lamina', 'pointer_operations.lamina',
    'pointer_operations_fixed.lamina', 'pointer_param_store.lamina',
    'primitive_types_comprehensive.lamina', 'primitive_types_fixed.lamina',
    'print_float.lamina', 'print_unsigned.lamina', 'readbyte_nul.lamina',
    'reassign_cast_loop.lamina', 'reassign_float_unary_self.lamina',
    'recursion_with_calculation.lamina', 'recursive_factorial.lamina',
    'recursive_fibonacci.lamina', 'recursive_sum.lamina',
    'register_pressure.lamina', 'register_size_mismatch_bug_test.lamina',
    'select_all_types.lamina', 'simple_const.lamina', 'simple_heap_test.lamina',
    'sroa_refusals.lamina', 'stack_align.lamina', 'stack_allocation.lamina',
    'stack_frame_overflow_bug_test.lamina', 'stack_operations.lamina',
    'stack_overflow_test.lamina', 'stdin.lamina', 'stress_test.lamina',
    'struct_field_offset_bug_test.lamina',
    'struct_operations_comprehensive.lamina', 'switch_dispatch.lamina',
    'tail_call_callee_saved.lamina', 'tail_call_float_args.lamina',
    'tail_stack_address.lamina', 'test_minimal.lamina', 'test_u8.lamina',
    'tiny.lamina', 'type_conversions_fixed.lamina', 'unary_ops.lamina',
    'unreachable_terminator.lamina', 'variables.lamina',
    'variadic_wide_argument.lamina', 'void_tail_print.lamina',
    'volatile_access.lamina',
}

def jit_host_problem():
    """Why the JIT mode cannot run on this host, or None."""
    if platform.system() == 'Linux' and platform.machine() in ('x86_64', 'AMD64'):
        return None
    return 'the JIT mode runs on x86_64 Linux only'

def compile_and_run_jit(test_path):
    """Run a testcase through `lamina --jit` and return (success, output_lines, stderr).

    LAMINA_JIT_NO_FALLBACK makes a failed in-memory compile an error. Without it the
    CLI quietly runs the AOT build instead, and this mode would pass for the wrong reason.
    """
    project_root = Path(__file__).parent
    test_path = Path(test_path)
    env = dict(os.environ, LAMINA_JIT_NO_FALLBACK='1')
    argv = ['cargo', 'run', '--release', '--quiet', '--', str(test_path), '--jit']
    if test_path.name == 'io_types.lamina':
        success, raw = run_raw_output(argv, project_root, env)
        return (True, raw, '') if success else (False, raw, '')
    success, stdout, stderr, exit_code = run_program(
        argv, project_root, STDIN_INPUTS.get(test_path.name), 300, env
    )
    if is_crash(exit_code):
        return False, execution_failure(stdout, stderr, exit_code), ''
    output_lines = [line.strip() for line in stdout.split('\n') if line.strip()]
    return True, output_lines, stderr

def compile_and_run_test(test_path, use_mir=False, target=None, opt_level=None):
    """Compile and run a test case, return the output lines"""
    project_root = Path(__file__).parent
    test_path = Path(test_path)
    executable_name = test_path.stem
    # lamina names a PE output `<stem>.exe` whatever the host, so the suffix
    # follows the target, not `os.name`.
    if (target or '').endswith('_windows') or (target is None and os.name == 'nt'):
        executable_name += '.exe'
    # Built programs go under target/, which is ignored, so a run never leaves
    # binaries beside the sources or overwrites a tracked file.
    output_dir = Path(project_root) / 'target' / 'testcases'
    output_dir.mkdir(parents=True, exist_ok=True)
    executable_path = str((output_dir / executable_name).resolve())
    
    # Compile the test
    cmd_flags = "--emit-asm" if use_mir else ""
    target_flag = f"--target {target}" if target else ""
    opt_flag = f"--opt-level {opt_level}" if opt_level is not None else ""
    source_path = testcase_source(test_path, target or host_variant())
    compile_cmd = (
        f'cargo run --release --quiet -- {source_path} {cmd_flags} {target_flag} '
        f'{opt_flag} -o "{executable_path}"'
    )
    success, stdout, stderr, _ = run_command(compile_cmd, cwd=project_root)

    if not success:
        return False, f"Compilation failed: {stderr}"

    argv = execution_argv(executable_path, target)

    if test_path.name == 'io_types.lamina':
        return run_raw_output(argv, project_root)

    stdin_input = STDIN_INPUTS.get(test_path.name)
    _, stdout, stderr, exit_code = run_program(
        argv, project_root, stdin_input, run_timeout(target)
    )

    if is_crash(exit_code):
        record_crash(executable_path, exit_code)
        return False, execution_failure(stdout, stderr, exit_code)

    # Return output lines (filter out empty lines and debug text)
    output_lines = [line.strip() for line in stdout.split('\n') if line.strip()]

    return True, output_lines

def run_tests(use_mir=False, target=None, report_only=False, opt_level=None, jit=False):
    crashes = Path(__file__).parent / 'target' / 'crashes.txt'
    try:
        crashes.parent.mkdir(parents=True, exist_ok=True)
        crashes.write_text('')
    except OSError:
        pass
    """Run all test cases and report results"""
    mode = "MIR assembly" if use_mir else "default"
    if jit:
        problem = jit_host_problem()
        if problem:
            print_colored(f"⏭️  Skipping JIT run: {problem}", Colors.YELLOW)
            return 0
        mode = "JIT"
    if target:
        problem = cross_toolchain_problem(target)
        if problem:
            print_colored(f"⏭️  Skipping {target} run: {problem}", Colors.YELLOW)
            print_colored("   Install the cross gcc and qemu-user, or drop --target.", Colors.YELLOW)
            return 0
        mode = f"{mode}, target {target}"
    byteorder = CROSS_TARGETS[target]['byteorder'] if target else sys.byteorder

    print_colored(f"🧪 Running Lamina Test Suite ({mode})", Colors.BOLD + Colors.BLUE)
    print_colored("=" * 50, Colors.BLUE)

    passed = 0
    failed = 0
    skipped = 0
    failures = []

    # Discover tests
    test_files = discover("testcases/*.lamina")
    benchmark_files = discover("benchmarks/*/*.lamina")
    all_tests = test_files + benchmark_files
    if jit:
        all_tests = [p for p in all_tests if Path(p).name in JIT_TESTCASES]
        print_colored(f"Running {len(all_tests)} testcases under --jit", Colors.BLUE)
        if not all_tests:
            print_colored("No testcases are listed in JIT_TESTCASES yet", Colors.YELLOW)
            return 0

    for test_path in all_tests:
        test_name = Path(test_path).name
        print(f"\n📝 Testing {test_name}...", end="", flush=True)

        unsupported = CROSS_TARGETS[target]['unsupported'] if target else {}
        if Path(test_path).stem in unsupported:
            reason = unsupported[Path(test_path).stem]
            print_colored(f"\r📝 Testing {test_name}... ⏭️  SKIPPED (unsupported on {target}: {reason})", Colors.YELLOW)
            skipped += 1
            continue

        expected_output = load_expected_output(test_path, byteorder, target)
        if expected_output is None:
            print_colored(f"\n❌ FAILED {test_name}: No expected output found", Colors.RED)
            failures.append((test_name, "No expected output found"))
            failed += 1
            continue

        if jit:
            success, result, stderr = compile_and_run_jit(test_path)
        else:
            success, result = compile_and_run_test(test_path, use_mir, target, opt_level)
            stderr = ''
        
        if not success:
            print_colored(f"\n❌ FAILED {test_name}: {result}", Colors.RED)
            failures.append((test_name, result))
            failed += 1
            continue
        
        actual_output = result
        
        if actual_output == expected_output:
            print_colored(f"\r📝 Testing {test_name}... ✅ PASSED", Colors.GREEN)
            # print(f"   Output: {actual_output}")
            passed += 1
        else:
            print_colored(f"\n❌ FAILED {test_name}: Output mismatch", Colors.RED)
            print(f"   Expected: {expected_output}")
            print(f"   Actual:   {actual_output}")
            reason = f"Output mismatch; expected {expected_output}, got {actual_output}"
            if stderr:
                print(f"   stderr:   {stderr}")
                reason += f"; stderr: {stderr}"
            failures.append((test_name, reason))
            failed += 1
    
    # Summary
    print_colored("\n" + "=" * 50, Colors.BLUE)
    total = passed + failed
    if skipped:
        print_colored(f"⏭️  {skipped} testcases skipped as unsupported on {target}", Colors.YELLOW)
    if failed == 0:
        print_colored(f"🎉 All {total} tests PASSED!", Colors.GREEN + Colors.BOLD)
    else:
        print_colored(f"📊 Results: {passed}/{total} passed, {failed} failed", Colors.YELLOW)
        # Repeated at the end so CI logs that elide the middle still show them.
        print_colored("Failed tests:", Colors.RED + Colors.BOLD)
        for name, reason in failures:
            print_colored(f"  {name}: {reason}", Colors.RED)
    return failed

def run_levels(use_mir=False, target=None, report_only=False, opt_level=None, jit=False):
    """Run the suite once per requested level and return the process exit code."""
    if opt_level is None:
        levels = [None]
    elif str(opt_level) == 'all':
        levels = [0, 1, 2, 3]
    else:
        levels = [int(opt_level)]
    if jit:
        return 1 if run_tests(jit=True) and not report_only else 0
    failed = sum(run_tests(use_mir, target, report_only, level) for level in levels)
    return 1 if failed and not report_only else 0

def list_tests():
    """List available test cases"""
    print_colored("📋 Available Test Cases:", Colors.BOLD + Colors.BLUE)
    
    test_files = discover("testcases/*.lamina")
    benchmark_files = discover("benchmarks/*/*.lamina")
    
    print_colored("\nStandard Tests:", Colors.BLUE)
    for test_path in test_files:
        print(f"  {Path(test_path).name}")
        
    print_colored("\nBenchmarks:", Colors.BLUE)
    for test_path in benchmark_files:
        print(f"  {Path(test_path).name} ({test_path})")

def run_single_test(test_name, use_mir=False, target=None):
    """Run a single test case"""
    # Try to find the test file
    if os.path.exists(test_name):
        test_path = test_name
    elif os.path.exists(f"testcases/{test_name}"):
        test_path = f"testcases/{test_name}"
    else:
        # Search in benchmarks
        matches = glob.glob(f"benchmarks/*/{test_name}")
        if matches:
            test_path = matches[0]
        else:
            print_colored(f"❌ Test '{test_name}' not found", Colors.RED)
            list_tests()
            return
    
    if target:
        problem = cross_toolchain_problem(target)
        if problem:
            print_colored(f"⏭️  Skipping {target} run: {problem}", Colors.YELLOW)
            return

    print_colored(f"🧪 Running single test: {test_path}", Colors.BOLD + Colors.BLUE)
    
    byteorder = CROSS_TARGETS[target]['byteorder'] if target else sys.byteorder
    expected_output = load_expected_output(test_path, byteorder, target)
    if expected_output is None:
        print_colored(f"⚠️  No expected output found for {test_path}", Colors.YELLOW)
        return

    unsupported = CROSS_TARGETS[target]['unsupported'] if target else {}
    if Path(test_path).stem in unsupported:
        print_colored(f"⏭️  SKIPPED (unsupported on {target}: {unsupported[Path(test_path).stem]})", Colors.YELLOW)
        return

    success, result = compile_and_run_test(test_path, use_mir, target)
    
    if not success:
        print_colored(f"❌ FAILED: {result}", Colors.RED)
        return
    
    actual_output = result
    
    if actual_output == expected_output:
        print_colored(f"✅ PASSED", Colors.GREEN)
        print(f"   Output: {actual_output}")
    else:
        print_colored(f"❌ FAILED: Output mismatch", Colors.RED)
        print(f"   Expected: {expected_output}")
        print(f"   Actual:   {actual_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lamina Test Runner')
    parser.add_argument('test', nargs='?', help='Specific test case to run')
    parser.add_argument('--list', action='store_true', help='List available tests')
    parser.add_argument('--mir', action='store_true', help='Use MIR codegen backend')
    parser.add_argument('--target', choices=sorted(CROSS_TARGETS),
                        help='Compile for a cross target and run under its emulator; '
                             'skipped with a message when the toolchain is absent')
    parser.add_argument('--opt-level', choices=['0', '1', '2', '3', 'all'],
                        help='Compile at this optimization level instead of the CLI '
                             'default; without it the suite never executes code built '
                             'by sroa, mem2reg, peephole, unrolling, CSE, the scheduler '
                             'or rotation. "all" runs the whole suite once per level')
    parser.add_argument('--jit', action='store_true',
                        help='Run the JIT-capable testcases through `lamina --jit` on the host '
                             'and check their AOT output (x86_64 Linux only)')
    parser.add_argument('--report-only', action='store_true',
                        help='Print the counts and exit 0 even when testcases fail, '
                             'for platforms whose backend is still incomplete')
    
    args = parser.parse_args()
    
    if args.jit and (args.mir or args.target or args.opt_level or args.test):
        parser.error('--jit runs the JIT testcase set at the default optimization level; '
                     'drop --mir, --target, --opt-level and the test name')

    if args.list:
        list_tests()
    elif args.test:
        run_single_test(args.test, args.mir, args.target)
    else:
        sys.exit(run_levels(args.mir, args.target, args.report_only, args.opt_level, args.jit))
