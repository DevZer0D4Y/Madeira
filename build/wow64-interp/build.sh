#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Madeira Converter Exception: see LICENSE-EXCEPTION.md
#
# Builds wow64interp.dll, the WoW64 CPU module that runs 32-bit x86 code with
# X64Interpreter instead of a JIT (cpu.c, docs/NO-JIT.md):
#   OUT/x86_64/wow64interp.dll   with mingw-w64 GCC: for Wine on an x86-64 PC,
#                                in place of wow64cpu.dll (a test bed)
#   OUT/aarch64/wow64interp.dll  with clang and lld: for Madeira, in place of
#                                FEX's xtajit.dll
# Either is skipped, with a note, when its compiler is missing. The module
# imports only ntdll and wow64; the import libraries come from the .def files.
#
# Usage: build.sh [OUT]   (default: build/wow64-interp/out)

set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:-$here/out}
sources=("$here/cpu.c" "$root/app/Madeira/X64Interpreter.c")
# No libm beyond ntdll's exports: GCC must not merge sin and cos into sincos.
cflags=(-std=gnu11 -O2 -Wall -Wextra -fno-math-errno -fno-builtin-sin -fno-builtin-cos -I "$root/app/Madeira")
built=0

if command -v x86_64-w64-mingw32-gcc >/dev/null && command -v x86_64-w64-mingw32-dlltool >/dev/null; then
    dir=$out/x86_64
    mkdir -p "$dir"
    x86_64-w64-mingw32-dlltool -d "$here/ntdll-imports.def" -l "$dir/libntdll.a"
    x86_64-w64-mingw32-dlltool -d "$here/wow64-imports.def" -l "$dir/libwow64.a"
    # -Wno-array-bounds: a false positive in mingw-w64's NtCurrentTeb.
    x86_64-w64-mingw32-gcc "${cflags[@]}" -Wno-array-bounds -mcx16 -mpopcnt -shared -nostdlib -Wl,--entry,DllMain \
        -o "$dir/wow64interp.dll" "${sources[@]}" "$here/wow64interp.def" -L"$dir" -lntdll -lwow64 -lgcc
    echo "$dir/wow64interp.dll"
    built=$((built + 1))
else
    echo "note: no x86_64-w64-mingw32-gcc: the x86-64 module was not built" >&2
fi

if command -v clang >/dev/null && command -v ld.lld >/dev/null && command -v llvm-dlltool >/dev/null; then
    dir=$out/aarch64
    mkdir -p "$dir"
    llvm-dlltool -m arm64 -d "$here/ntdll-imports.def" -l "$dir/libntdll.a"
    llvm-dlltool -m arm64 -d "$here/wow64-imports.def" -l "$dir/libwow64.a"
    headers=()
    [ -d /usr/share/mingw-w64/include ] && headers=(-isystem /usr/share/mingw-w64/include)
    clang --target=aarch64-w64-mingw32 "${headers[@]}" "${cflags[@]}" -shared -nostdlib -fuse-ld=lld \
        -Wl,--entry,DllMain -o "$dir/wow64interp.dll" "${sources[@]}" "$here/wow64interp.def" -L"$dir" -lntdll -lwow64
    echo "$dir/wow64interp.dll"
    built=$((built + 1))
else
    echo "note: no clang, ld.lld and llvm-dlltool: the ARM64 module was not built" >&2
fi

[ "$built" -gt 0 ]
