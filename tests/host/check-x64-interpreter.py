#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Madeira Converter Exception: see LICENSE-EXCEPTION.md
"""No JIT: the x86-64 interpreter (app/Madeira/X64Interpreter.c) and its engine check.

Everywhere (Linux or macOS, ARM64 or x86-64):
  - the interpreter and app/Madeira/NoJITSelfCheck.c build with warnings as errors, and
    the engine check passes on this machine: each program in NoJITSelfCheckData.h (64-bit,
    and 32-bit with x87 and MMX) gives the result a real x86 CPU gave. On an ARM64 Mac
    that is the iPhone's architecture.
On x86-64 Linux, also:
  - the committed programs, run natively, still give the expected results recorded in
    NoJITSelfCheckData.h (the 32-bit ones when the compiler can build -m32);
  - the instruction fuzzer: about 1,400 64-bit snippets, each run natively and through
    the interpreter from the same random registers, flags, vectors, MXCSR and memory;
  - with -m32: about 800 32-bit snippets (integer code, x87, MMX) run natively in a
    32-bit process, recorded, and replayed through the interpreter's 32-bit mode.
With qemu-aarch64 and an aarch64-linux-gnu toolchain, also:
  - the engine check, and replays of the recorded native fuzz results, on ARM64.
With mingw-w64 GCC, or clang with lld and llvm-dlltool:
  - the WoW64 CPU module (build/wow64-interp) builds, imports nothing but ntdll and
    wow64, and exports every BTCpu entry point wow64.dll looks up.
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
# fuzz-common.h is shared by programs that use only part of it.
flags_fuzz = ['-Wno-unused-function']


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
        m32 = run(compiler, '-m32', '-x', 'c', '-', '-o', folder / 'm32-probe', input='int main(void){return 0;}\n').returncode == 0 \
            and run(folder / 'm32-probe').returncode == 0

        # The expected results are what an x86 CPU computes from the committed bytes.
        header = (app / 'NoJITSelfCheckData.h').read_text()
        for bits in (64, 32):
            if bits == 32 and not m32:
                print('NOTE: the compiler cannot build -m32: the 32-bit expected results were not run natively')
                continue
            start = header.index(f'nojit_selfcheck_blob{bits}[')
            body = header[start:header.index('};', start)]
            (folder / f'blob{bits}.bin').write_bytes(bytes(int(b, 16) for b in re.findall(r'0x([0-9a-f]{2}),', body)))
            cases = re.findall(r'\{ "(\w+)", ' + str(bits) + r', 0x([0-9a-f]+), 0x([0-9a-f]+)ull, 0x([0-9a-f]+)ull \}', header)
            runner = folder / f'run-native{bits}'
            built = run(compiler, *(['-m32'] if bits == 32 else []), '-O1', '-I', app, tests / 'run-native.c', '-o', runner)
            native = run(runner, folder / f'blob{bits}.bin', *[f'{n}:{o}:{s}' for n, o, s, _ in cases], timeout=300) \
                if built.returncode == 0 else built
            got = [line.split()[2] for line in native.stdout.splitlines()]
            want = [f'{int(e, 16):x}' for _, _, _, e in cases]
            check(native.returncode == 0 and got == want and len(cases) >= 20,
                  f'{len(cases)} expected {bits}-bit results in NoJITSelfCheckData.h match this x86 CPU',
                  native.stderr + str([(c[0], g, w) for c, g, w in zip(cases, got, want) if g != w]))

        # Instruction by instruction, against the CPU.
        made = run(sys.executable, tests / 'gen-fuzz-insns.py', folder / 'insns.S')
        assembled = run(compiler, '-c', folder / 'insns.S', '-o', folder / 'insns.o') if made.returncode == 0 else made
        fuzz = folder / 'fuzz'
        built = run(compiler, *flags, *flags_fuzz, '-std=gnu11', '-I', tests, tests / 'fuzz.c', app / 'X64Interpreter.c', folder / 'insns.o',
                    '-o', fuzz, *libs) if assembled.returncode == 0 else assembled
        check(built.returncode == 0, 'instruction fuzzer builds', built.stderr)
        recordings = []
        if built.returncode == 0:
            result = run(fuzz, 'run', 100, '0x5eed', timeout=900)
            check(result.returncode == 0, 'fuzzer, native against interpreted: ' + result.stdout.strip(),
                  '\n'.join(result.stderr.splitlines()[:60]))
            if m32:
                made = run(sys.executable, tests / 'gen-fuzz-insns32.py', folder / 'insns32.S')
                assembled = run(compiler, '-m32', '-c', folder / 'insns32.S', '-o', folder / 'insns32.o') if made.returncode == 0 else made
                fuzz32 = folder / 'fuzz32'
                built32 = run(compiler, '-m32', '-std=gnu11', '-O1', '-I', app, '-I', tests, tests / 'fuzz32.c', folder / 'insns32.o',
                              '-o', fuzz32, '-lm') if assembled.returncode == 0 else assembled
                rec32 = folder / 'rec32.bin'
                recorded = run(fuzz32, rec32, 100, '0x3232', timeout=900) if built32.returncode == 0 else built32
                check(recorded.returncode == 0, '32-bit snippets recorded natively: ' + recorded.stdout.strip(),
                      recorded.stderr[-3000:])
                if recorded.returncode == 0:
                    result = run(fuzz, 'replay', rec32, timeout=900)
                    check(result.returncode == 0, '32-bit, native against interpreted: ' + result.stdout.strip(),
                          '\n'.join(result.stderr.splitlines()[:60]))
                    recordings.append(rec32)
            else:
                print('NOTE: the compiler cannot build -m32: the 32-bit fuzzer was not run')

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
                    built = run(*cc, '-std=gnu11', '-O2', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function', '-I', app, '-I', tests,
                                '-c', source, '-o', obj)
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
                    for recording in [vectors] + recordings:
                        result = run(qemu, arm_fuzz, 'replay', recording, timeout=900)
                        check(result.returncode == 0, 'recorded native results on ARM64 (QEMU): ' + result.stdout.strip(),
                              '\n'.join(result.stderr.splitlines()[:60]))
            else:
                print('NOTE: no qemu-aarch64 and aarch64-linux-gnu-gcc: the ARM64 replay was not run')
    else:
        print('NOTE: not x86-64 Linux: the native comparisons were not run')

    # The WoW64 CPU module: what it imports and exports.
    objdump = shutil.which('llvm-objdump') or shutil.which('x86_64-w64-mingw32-objdump')
    if objdump and (shutil.which('x86_64-w64-mingw32-gcc') or (shutil.which('clang') and shutil.which('ld.lld'))):
        built = run('bash', root / 'build/wow64-interp/build.sh', folder / 'wow64interp', timeout=600)
        dlls = [Path(line) for line in built.stdout.splitlines() if line.endswith('.dll')]
        check(built.returncode == 0 and dlls, 'the WoW64 CPU module builds: ' + ', '.join(d.parent.name for d in dlls),
              built.stderr[-3000:])
        required = ['BTCpuGetBopCode', 'BTCpuGetContext', 'BTCpuIsProcessorFeaturePresent', 'BTCpuProcessInit',
                    'BTCpuResetToConsistentState', 'BTCpuSetContext', 'BTCpuSimulate', '__wine_get_unix_opcode']
        for dll in dlls:
            dump = run(objdump, '-p', dll).stdout
            imports = sorted(set(re.findall(r'DLL Name: (\S+)', dump)))
            missing = [name for name in required if not re.search(r'\b' + re.escape(name) + r'\b', dump)]
            check(imports == ['ntdll.dll', 'wow64.dll'] and not missing,
                  f'{dll.parent.name} module imports {", ".join(imports)} only and exports the BTCpu entry points',
                  f'imports {imports}, missing exports {missing}')
    else:
        print('NOTE: no mingw-w64 GCC or clang with lld: the WoW64 CPU module was not built')

sys.exit(1 if failures else 0)
