#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Madeira Converter Exception: see LICENSE-EXCEPTION.md
#
# Builds wow64interp.dll, the WoW64 CPU module that runs 32-bit x86 code with
# X64Interpreter instead of a JIT (cpu.c, docs/NO-JIT.md):
#   OUT/x86_64/wow64interp.dll   with mingw-w64 GCC: for Wine on an x86-64 PC,
#                                in place of wow64cpu.dll (a test bed)
#   OUT/aarch64/wow64interp.dll  for Madeira, beside FEX's xtajit.dll: with
#                                llvm-mingw's aarch64-w64-mingw32-clang when it is
#                                on PATH (as build/windows-runtime/build.sh puts it),
#                                else with clang, lld and the system's mingw headers
# Either is skipped, with a note, when its compiler is missing. The module
# imports only ntdll and wow64; the import libraries come from the .def files.
#
# Usage: build.sh [--install] [OUT]   (default OUT: build/wow64-interp/out)
#   --install  also copy the ARM64 module, stripped, to
#              app/Madeira/aarch64-windows/wow64interp.dll (the madeira.cfg key
#              wow-cpu = interpreter selects it); fails without an ARM64 build

set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
install=0
if [ "${1:-}" = --install ]; then install=1; shift; fi
out=${1:-$here/out}
sources=("$here/cpu.c" "$root/app/Madeira/X64Interpreter.c")
# No libm beyond ntdll's exports: GCC must not merge sin and cos into sincos.
cflags=(-std=gnu11 -O2 -Wall -Wextra -fno-math-errno -fno-builtin-sin -fno-builtin-cos -I "$root/app/Madeira")
built=0

if [ "$install" = 1 ]; then
    :   # Madeira needs the ARM64 module only
elif command -v x86_64-w64-mingw32-gcc >/dev/null && command -v x86_64-w64-mingw32-dlltool >/dev/null; then
    dir=$out/x86_64
    mkdir -p "$dir"
    x86_64-w64-mingw32-dlltool -d "$here/ntdll-imports.def" -l "$dir/libntdll.a"
    x86_64-w64-mingw32-dlltool -d "$here/wow64-imports.def" -l "$dir/libwow64.a"
    # -Wno-array-bounds: a false positive in mingw-w64's NtCurrentTeb. The compiler's own
    # runtime (libgcc, or compiler-rt under llvm-mingw's gcc wrapper) for the stack probe.
    x86_64-w64-mingw32-gcc "${cflags[@]}" -Wno-array-bounds -mcx16 -mpopcnt -shared -nostdlib -Wl,--entry,DllMain \
        -o "$dir/wow64interp.dll" "${sources[@]}" "$here/wow64interp.def" -L"$dir" -lntdll -lwow64 \
        "$(x86_64-w64-mingw32-gcc -print-libgcc-file-name)"
    echo "$dir/wow64interp.dll"
    built=$((built + 1))
else
    echo "note: no x86_64-w64-mingw32-gcc: the x86-64 module was not built" >&2
fi

arm64=()
if command -v aarch64-w64-mingw32-clang >/dev/null; then
    arm64=(aarch64-w64-mingw32-clang)                       # llvm-mingw: its own headers and lld
elif command -v clang >/dev/null && command -v ld.lld >/dev/null; then
    arm64=(clang --target=aarch64-w64-mingw32 -fuse-ld=lld)
    [ -d /usr/share/mingw-w64/include ] && arm64+=(-isystem /usr/share/mingw-w64/include)
fi
if [ ${#arm64[@]} -gt 0 ] && command -v llvm-dlltool >/dev/null; then
    dir=$out/aarch64
    mkdir -p "$dir"
    llvm-dlltool -m arm64 -d "$here/ntdll-imports.def" -l "$dir/libntdll.a"
    llvm-dlltool -m arm64 -d "$here/wow64-imports.def" -l "$dir/libwow64.a"
    # 64 KiB sections, linked as winegcc links Wine's own ARM64 modules: code and data on
    # separate 16 KiB pages, so the module can be wrapped as a signed Mach-O image later
    # (docs/NO-JIT.md, step 4).
    "${arm64[@]}" "${cflags[@]}" -shared -nostdlib -Wl,--entry,DllMain -Wl,--no-insert-timestamp \
        -Wl,-Xlink=-filealign:0x1000,-Xlink=-align:0x10000,-Xlink=-driver \
        -o "$dir/wow64interp.dll" "${sources[@]}" "$here/wow64interp.def" -L"$dir" -lntdll -lwow64
    echo "$dir/wow64interp.dll"
    built=$((built + 1))
    if [ "$install" = 1 ]; then
        strip=$(command -v aarch64-w64-mingw32-strip || command -v llvm-strip)
        "$strip" --strip-debug -o "$root/app/Madeira/aarch64-windows/wow64interp.dll" "$dir/wow64interp.dll"
        echo "installed $root/app/Madeira/aarch64-windows/wow64interp.dll"
    fi
else
    echo "note: no aarch64-w64-mingw32-clang, or clang with lld, and llvm-dlltool: the ARM64 module was not built" >&2
    [ "$install" = 0 ] || { echo "error: --install needs the ARM64 module" >&2; exit 1; }
fi

[ "$built" -gt 0 ]
