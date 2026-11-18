#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Madeira Converter Exception: see LICENSE-EXCEPTION.md
"""No JIT: the WoW64 CPU module (build/wow64-interp) running 32-bit programs in Wine.

Needs a Wine built for x86-64 Linux with its new WoW64 (configure
--enable-archs=i386,x86_64; Madeira's own Wine works, with build/ next to its
source as in this repository), mingw-w64 GCC for x86-64 and i686, and an
x86-64 Linux machine. Not part of CI: building Wine takes about an hour.

Usage: check-wow64interp-wine.py WINE_BUILD_DIR [--prefix DIR] [--keep]

Makes a Wine prefix, installs wow64interp.dll in system32, and runs two 32-bit
programs from tests/x64-interp/wine twice: with Wine's wow64cpu.dll, which runs
them on this CPU, and with wow64interp.dll (HKLM\\Software\\Microsoft\\Wow64\\x86
names the module wow64.dll loads). Every output line not starting with "info:"
must be the same (lines starting with "approx" to 1e-13: transcendental results,
which the x87 does not round correctly), and the second run must have loaded
wow64interp.dll:
  - wowtest.exe: exceptions from the CPU (access violations, divide error,
    INT3, UD2, INTO, stack overflow) caught and resumed from a changed context,
    RaiseException, threads with atomics, x87 math through msvcrt, 64-bit
    division, file I/O, sockets (unix calls);
  - workloads.exe: the engine check's 32-bit programs (tests/x64-interp/workloads.c).
"""
from pathlib import Path
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

root = Path(__file__).resolve().parents[2]
tests = root / 'tests/x64-interp/wine'
failures = 0


def check(condition, label, detail=''):
    global failures
    print(('PASS: ' if condition else 'FAIL: ') + label, flush=True)
    if not condition:
        failures += 1
        if detail:
            print(detail.rstrip())


def run(*args, timeout=600, **kwargs):
    return subprocess.run([str(a) for a in args], capture_output=True, text=True, timeout=timeout, **kwargs)


parser = argparse.ArgumentParser()
parser.add_argument('wine_build', type=Path)
parser.add_argument('--prefix', type=Path)
parser.add_argument('--keep', action='store_true', help='keep the prefix and the programs')
options = parser.parse_args()

wine = options.wine_build.resolve() / 'wine'
wineserver = options.wine_build.resolve() / 'server/wineserver'
if not wine.exists() or not wineserver.exists():
    raise SystemExit(f'no wine or server/wineserver in {options.wine_build}')

work = Path(tempfile.mkdtemp(prefix='wow64interp-wine-'))
prefix = (options.prefix or work / 'prefix').resolve()
env = dict(os.environ, WINEPREFIX=str(prefix), WINEDEBUG='-all', WINEDLLOVERRIDES='mscoree,mshtml=',
           WINESERVER=str(wineserver), DISPLAY='')

try:
    # The module and the programs.
    built = run('bash', root / 'build/wow64-interp/build.sh', work / 'module')
    module = work / 'module/x86_64/wow64interp.dll'
    check(built.returncode == 0 and module.exists(), 'wow64interp.dll builds for x86-64', built.stderr)
    cc = 'i686-w64-mingw32-gcc'
    programs = {
        'wowtest': run(cc, '-O2', '-msse2', '-Wall', '-Wextra', '-Werror', tests / 'wowtest.c', '-o', work / 'wowtest.exe',
                       '-lws2_32'),
        # As the engine check builds them: x87 floating point, MMX.
        'workloads': run(cc, '-O2', '-march=i686', '-mmmx', '-msse', '-mfpmath=387', '-fno-tree-loop-distribute-patterns',
                         '-Wall', '-Wextra', '-Wno-unused-parameter', '-Werror', '-I', root / 'app/Madeira',
                         tests / 'workloads-main.c', '-o', work / 'workloads.exe'),
    }
    for name, result in programs.items():
        check(result.returncode == 0, f'{name}.exe builds (32-bit)', result.stderr)
    if failures:
        raise SystemExit(1)

    # A prefix with the module in system32.
    if not (prefix / 'system.reg').exists():
        started = time.time()
        made = run(wine, 'wineboot', '--init', env=env, timeout=1200)
        run(wineserver, '-w', env=env, timeout=600)
        check((prefix / 'system.reg').exists(), f'Wine prefix made in {time.time() - started:.0f} s', made.stderr[-3000:])
    system32 = prefix / 'drive_c/windows/system32'
    shutil.copy(module, system32 / 'wow64interp.dll')

    outputs = {}
    for cpu in ('wow64cpu.dll', 'wow64interp.dll'):
        added = run(wine, 'reg', 'add', r'HKLM\Software\Microsoft\Wow64\x86', '/ve', '/d', cpu, '/f', env=env)
        run(wineserver, '-w', env=env, timeout=600)
        check(added.returncode == 0, f'Wow64\\x86 set to {cpu}', added.stdout + added.stderr)
        for name in programs:
            started = time.time()
            # +loaddll names the modules each process loads, so the run shows which CPU module it had.
            result = run(wine, work / f'{name}.exe', env=dict(env, WINEDEBUG='-all,+loaddll'), timeout=1800)
            seconds = time.time() - started
            loaded = re.findall(r'Loaded L"[^"]*\\(wow64\w*\.dll)"', result.stderr, re.IGNORECASE)
            lines = result.stdout.replace('\r', '').splitlines()
            outputs[cpu, name] = lines
            check(result.returncode == 0 and cpu.lower() in [m.lower() for m in loaded],
                  f'{name}.exe with {cpu}: exit {result.returncode}, {len(lines)} lines, {seconds:.1f} s, CPU module '
                  f'{", ".join(sorted(set(m for m in loaded if m.lower() != "wow64.dll"))) or "none"}',
                  '\n'.join(result.stdout.splitlines()[-20:]) + '\n' +
                  '\n'.join(line for line in result.stderr.splitlines() if 'loaddll' not in line)[-4000:])
            for line in lines:
                if line.startswith('info:'):
                    print(f'      {cpu}: {line}')

    def same(a, b):
        if a == b:
            return True
        if not (a.startswith('approx ') and b.startswith('approx ')):
            return False
        x, y = a.split(), b.split()   # numbers to 1e-13
        if len(x) != len(y):
            return False
        for p, q in zip(x, y):
            try:
                u, v = float(p.rstrip(':')), float(q.rstrip(':'))
            except ValueError:
                if p != q:
                    return False
                continue
            if u != v and abs(u - v) > 1e-13 * max(abs(u), abs(v)):
                return False
        return True

    for name in programs:
        native = [line for line in outputs['wow64cpu.dll', name] if not line.startswith('info:')]
        interpreted = [line for line in outputs['wow64interp.dll', name] if not line.startswith('info:')]
        differences = [f'  wow64cpu:    {a}\n  wow64interp: {b}' for a, b in zip(native, interpreted) if not same(a, b)]
        if len(native) != len(interpreted):
            differences.append(f'  {len(native)} lines against {len(interpreted)}')
        check(native and not differences and native[-1] == ('done' if name == 'wowtest' else native[-1]),
              f'{name}.exe: the same {len(native)} lines from the interpreter as from the CPU', '\n'.join(differences[:40]))
finally:
    run(wineserver, '-k', env=env, timeout=120)
    if options.keep:
        print(f'kept {work}')
    else:
        shutil.rmtree(work, ignore_errors=True)

sys.exit(1 if failures else 0)
