#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Madeira Converter Exception: see LICENSE-EXCEPTION.md
"""No JIT: the x86-64 interpreter (app/Madeira/X64Interpreter.c) and its engine check.

Everywhere (Linux or macOS, ARM64 or x86-64):
  - the interpreter and app/Madeira/NoJITSelfCheck.c build with warnings as errors, and
    the engine check passes on this machine: each program in NoJITSelfCheckData.h gives
    the result a real x86-64 CPU gave. On an ARM64 Mac that is the iPhone's architecture.
On x86-64 Linux, also:
  - the committed programs, run natively, still give the expected results recorded in
    NoJITSelfCheckData.h;
  - the instruction fuzzer: about 1,400 snippets, each run natively and through the
    interpreter from the same random registers, flags, vectors, MXCSR and memory.
With qemu-aarch64 and an aarch64-linux-gnu toolchain, also:
  - the engine check, and a replay of recorded native fuzz results, on ARM64.
"""
from pathlib import Path
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[2]
app = root / 'app/Madeira'
tests = root / 'tests/x64-interp'
failures = 0


def check(condition, label, detail=''):
    global failures
    print(('PASS: ' if condition else 'FAIL: ') + label)
    if not condition:
        failures += 1
        if detail:
            print(detail.rstrip())


def run(*args, **kwargs):
    return subprocess.run([str(a) for a in args], capture_output=True, text=True, **kwargs)


machine = platform.machine().lower()
x86 = machine in ('x86_64', 'amd64')
compiler = os.environ.get('CC') or shutil.which('clang') or shutil.which('cc') or shutil.which('gcc')
if not compiler:
    raise SystemExit('No C compiler')
is_gcc = 'clang' not in run(compiler, '--version').stdout.lower()
flags = ['-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-I', app]
if is_gcc:
    flags += ['-frounding-math']   # clang gets this from the file's FENV_ACCESS pragma
if x86:
    flags += ['-mcx16']
libs = ['-lm'] + (['-latomic'] if is_gcc else [])

with tempfile.TemporaryDirectory(prefix='x64-interp-') as folder:
    folder = Path(folder)

    # The engine check, as the app runs it, on this machine.
    selfcheck = folder / 'selfcheck'
    built = run(compiler, *flags, tests / 'selfcheck-host.c', app / 'NoJITSelfCheck.c', app / 'X64Interpreter.c',
                '-o', selfcheck, *libs)
    check(built.returncode == 0, f'interpreter and engine check build with -Werror ({Path(compiler).name}, {machine})',
          built.stderr)
    if built.returncode == 0:
        result = run(selfcheck, timeout=600)
        summary = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ''
        check(result.returncode == 0, f'engine check on {machine}: {summary}', result.stdout + result.stderr)

    if x86 and platform.system() == 'Linux':
        # The expected results are what an x86-64 CPU computes from the committed bytes.
        header = (app / 'NoJITSelfCheckData.h').read_text()
        body = header[header.index('nojit_selfcheck_blob['):header.index('};')]
        (folder / 'blob.bin').write_bytes(bytes(int(b, 16) for b in re.findall(r'0x([0-9a-f]{2}),', body)))
        cases = re.findall(r'\{ "(\w+)", 0x([0-9a-f]+), 0x([0-9a-f]+)ull, 0x([0-9a-f]+)ull \}', header)
        runner = folder / 'run-native'
        built = run(compiler, '-O1', '-I', app, tests / 'run-native.c', '-o', runner)
        native = run(runner, folder / 'blob.bin', *[f'{n}:{o}:{s}' for n, o, s, _ in cases], timeout=300) \
            if built.returncode == 0 else built
        got = [line.split()[2] for line in native.stdout.splitlines()]
        want = [f'{int(e, 16):x}' for _, _, _, e in cases]
        check(native.returncode == 0 and got == want and len(cases) >= 20,
              f'{len(cases)} expected results in NoJITSelfCheckData.h match this x86-64 CPU',
              native.stderr + str([(c[0], g, w) for c, g, w in zip(cases, got, want) if g != w]))

        # Instruction by instruction, against the CPU.
        made = run(sys.executable, tests / 'gen-fuzz-insns.py', folder / 'insns.S')
        assembled = run(compiler, '-c', folder / 'insns.S', '-o', folder / 'insns.o') if made.returncode == 0 else made
        fuzz = folder / 'fuzz'
        built = run(compiler, *flags, '-std=gnu11', tests / 'fuzz.c', app / 'X64Interpreter.c', folder / 'insns.o',
                    '-o', fuzz, *libs) if assembled.returncode == 0 else assembled
        check(built.returncode == 0, 'instruction fuzzer builds', built.stderr)
        if built.returncode == 0:
            result = run(fuzz, 'run', 100, '0x5eed', timeout=900)
            check(result.returncode == 0, 'fuzzer, native against interpreted: ' + result.stdout.strip(),
                  '\n'.join(result.stderr.splitlines()[:60]))

            # The same comparison on ARM64, when this machine can build and emulate it.
            qemu, cross = shutil.which('qemu-aarch64'), shutil.which('aarch64-linux-gnu-gcc')
            if qemu and cross:
                vectors = folder / 'vectors.bin'
                recorded = run(fuzz, 'record', vectors, 30, '0xa64', timeout=900)
                objects = []
                ok = recorded.returncode == 0
                cross_clang = shutil.which('clang') and Path('/usr/aarch64-linux-gnu/include').is_dir()
                for source in [tests / 'fuzz.c', tests / 'selfcheck-host.c', app / 'NoJITSelfCheck.c', app / 'X64Interpreter.c']:
                    obj = folder / ('arm64-' + source.stem + '.o')
                    cc = ['clang', '--target=aarch64-linux-gnu', '-isystem', '/usr/aarch64-linux-gnu/include'] if cross_clang \
                        else [cross, '-frounding-math']
                    built = run(*cc, '-std=gnu11', '-O2', '-Wall', '-Wextra', '-Werror', '-I', app, '-c', source, '-o', obj)
                    ok = ok and built.returncode == 0
                    objects.append(obj)
                arm_check = folder / 'selfcheck-arm64'
                arm_fuzz = folder / 'fuzz-arm64'
                link = ['-static', '-lm'] + ([] if cross_clang else ['-latomic'])
                ok = ok and run(cross, objects[1], objects[2], objects[3], '-o', arm_check, *link).returncode == 0
                ok = ok and run(cross, objects[0], objects[3], '-o', arm_fuzz, *link).returncode == 0
                check(ok, 'ARM64 builds of the engine check and the fuzz replay', recorded.stderr)
                if ok:
                    result = run(qemu, arm_check, timeout=900)
                    summary = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ''
                    check(result.returncode == 0, f'engine check on ARM64 (QEMU): {summary}', result.stdout + result.stderr)
                    result = run(qemu, arm_fuzz, 'replay', vectors, timeout=900)
                    check(result.returncode == 0, 'recorded x86-64 results on ARM64 (QEMU): ' + result.stdout.strip(),
                          '\n'.join(result.stderr.splitlines()[:60]))
            else:
                print('NOTE: no qemu-aarch64 and aarch64-linux-gnu-gcc: the ARM64 replay was not run')
    else:
        print('NOTE: not x86-64 Linux: the native comparisons were not run')

sys.exit(1 if failures else 0)
