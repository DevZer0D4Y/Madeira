# No JIT (experimental)

**Settings → JIT → No JIT (experimental)** is the start of a way to run
Windows games with no debugger attached: no StikDebug, no pairing file, no
LocalDevVPN. It does not start games yet. This page says what exists, how it
is tested, and what is left, starting with older 32-bit games, the closest.

## Why Madeira needs JIT

iOS lets a process write code and then run it only while a debugger is
attached (`CS_DEBUGGED`); [JIT setup](JIT.md) is how Madeira gets one. Two
things in a game session need that:

1. **The game's x86 code.** FEX translates it to ARM64 as it runs, writing the
   translation into memory and then executing it.
2. **Wine's own Windows libraries.** They are ARM64EC or ARM64 PE files, not
   Mach-O, so iOS cannot load them as signed code: Madeira copies their code
   into the JIT pool and runs it from there.

The interpreter replaces the first. The second is why games cannot start
without JIT yet.

## The way for 32-bit games

Older games are mostly 32-bit, and a 32-bit Windows process runs through
WoW64: all of its Windows code is 32-bit x86 (the game and Wine's i386 DLLs,
kernel32, user32, the D3D9 front end and the rest), except four native ARM64
modules on the 64-bit side:

| Module | What it is |
|---|---|
| `ntdll.dll` | the 64-bit ntdll |
| `wow64.dll`, `wow64win.dll` | Wine's WoW64 layer: 32-bit system calls to 64-bit ones |
| the CPU module | runs the 32-bit code: FEX's `xtajit.dll` today, `wow64interp.dll` (below) for No JIT |

The interpreter runs all of the 32-bit code as data, so no JIT is needed for
it. What remains is to run the four native modules without the JIT pool.
Unix calls (Metal through winemetal, audio, sockets) go straight to Madeira's
native side, which is part of the signed app; with `d3d9 = native`, DXMT's
D3D9 front end runs there too, so only the game's own code is interpreted.

Steps, in order:

1. **The CPU module** (done, not yet run in Wine): `build/wow64-interp`, below.
2. **Run it in Wine on a PC.** Build Wine for x86-64 Linux with its new WoW64
   (`--enable-archs=i386,x86_64`) and point
   `HKLM\Software\Microsoft\Wow64\x86` at `wow64interp.dll` (wow64.dll reads
   the module's name there; the default is `wow64cpu.dll`). 32-bit programs
   then run interpreted, which tests the module against real Wine.
3. **Run it in Madeira with JIT on**, in place of `xtajit.dll`, the same way.
   The ARM64 build reads the TEB through x18 like Madeira's other ARM64 modules
   and needs the same handling. Check what Madeira's `wow64.dll` expects when
   the guest window base B is not 0: the module passes host addresses (B + a)
   for a unix call's arguments and a fault's exception address.
4. **The four native modules as signed code**: convert them to Mach-O when
   Madeira is built, so the sideloader signs them with the app and iOS loads
   them like any other library. The open upstream pull request #497 does this
   for `ntdll`, `kernelbase` and `kernel32` (with the offline x18 rewriting) and
   runs test programs on an iPhone with JIT off.
5. **A 32-bit session that starts without the JIT pool**: skip
   `StikJITHelper`'s pool for No JIT, and have the loader use the signed copies.
6. **Speed**, below.

64-bit games need more: every one of Wine's ARM64EC modules as signed code
(416 files today), and an interpreter-backed version of FEX's ARM64EC module
(`libarm64ecfex.dll`, Wine's `BTCpu64` interface).

## What the option does now

- Madeira asks for no debugger. **Play** says that No JIT cannot start games
  yet instead of enabling JIT, and **Enable JIT** says that No JIT is selected.
  Choosing Automatic, StikDebug or Built-in StikJIT plays as before.
- **JIT setup** (with No JIT selected) has **Run engine check**. It runs 25
  small x86 programs, each with two inputs, through the interpreter from
  ordinary heap memory that is never executable, and compares each of the 50
  results with the one a real x86 CPU gave for the same bytes. It reports the
  instructions run, the rate, and whether JIT was off (`CS_DEBUGGED` clear)
  during the check. The log gets a `[no-jit]` line with the same result.

The programs are in `tests/x64-interp/workloads.c`, built twice. The 64-bit
build uses the Windows x64 calling convention: CRC-32, quicksort, a prime
sieve, 8- to 128-bit division and multiplication, double and float matrix
work, SSE2/SSE3 vector code, `REP` string instructions, jump tables and
function pointers, `LOCK`ed atomics including `CMPXCHG16B`, every MXCSR
rounding mode with flush-to-zero, and calls from x86 code into host code and
back. The 32-bit build is made as older games were (`-m32 -mfpmath=387`, x87
floating point, 64-bit division through libgcc's helpers) and adds MMX and an
x87 program with C casts through `FLDCW`/`FISTP` and 64-bit integers through
`FILD`/`FISTP`; it runs in a 4 MiB guest window, as Madeira's WoW64 does.

## The interpreter

`app/Madeira/X64Interpreter.c` decodes each instruction from guest memory and
carries it out in C. Nothing is generated or cached, so self-modifying code
and code a game's own JIT writes (Mono, .NET, LuaJIT) need nothing special.
All state is in one `X64State` per thread.

- **64-bit mode**: guest addresses are host addresses, as under FEX in Wine's
  ARM64EC process.
- **32-bit mode** (`x64_state_init32`): 32-bit registers, addresses, stack and
  EIP; guest address a is host address `guest_base + a` (Madeira's guest
  window B), and the instructions only 32-bit mode has (`INC`/`DEC` 40-4F,
  `PUSHA`/`POPA`, the BCD adjusts, segment register pushes and pops, `INTO`).

Implemented:

- the whole general-purpose instruction set, with x86's flags; `LOCK` prefixes
  as real host atomics, and `XCHG` with memory always atomic;
- string instructions with `REP`, `REPE` and `REPNE`, in both directions;
- **x87**, the whole instruction set: registers keep all 80 bits through loads,
  stores, moves, integer and BCD conversions and comparisons, and arithmetic is
  done in double, which is what Windows' default control word (0x27F, 53-bit
  precision) asks for. Rounding follows the control word, and empty and full
  registers behave as x87's masked stack faults do. Transcendental results
  (`FSIN`, `FPTAN`, `FYL2X`...) are double's. A result outside double's
  exponent range, which x87 can hold, becomes infinity or zero;
- **MMX**, on the x87 registers as the hardware aliases them, with the integer
  SSE additions to it (`PSHUFW`, `PSADBW`, `PMINUB`...) and the MMX conversions;
- SSE, SSE2, SSE3, `PSHUFB` and `PALIGNR`;
- `CMPXCHG8B`/`CMPXCHG16B`, `POPCNT`, `LZCNT`, `TZCNT`, `BSWAP`, `CPUID`,
  `RDTSC`, fences, `LDMXCSR`/`STMXCSR`, `FXSAVE`/`FXRSTOR`;
- FS and GS segment bases (GS is the 64-bit TEB on Windows, FS the 32-bit one).

SSE floating point follows x86 rather than the host: which NaN an operation
returns and the default NaN, MXCSR rounding, DAZ and FTZ, and the "integer
indefinite" result of an out-of-range conversion. On ARM64, aligned loads are
acquire and aligned stores release, which keeps x86's ordering of memory
between threads. `CPUID` describes what is implemented (x87, MMX, SSE to SSE3,
`CMPXCHG16B`, `POPCNT`, `LZCNT`; no AVX), so a program that checks before using
newer instructions picks a path the interpreter has.

Not implemented: AVX and anything VEX- or EVEX-encoded, SSSE3 beyond `PSHUFB`
and `PALIGNR`, SSE4, 16-bit addressing, far calls and jumps, and system
instructions. Any of these stops the run with `X64_EXIT_UNSUPPORTED` (or
`X64_EXIT_PRIVILEGED`) before the instruction changes anything, with the
instruction's address and bytes, so the caller can report it. MXCSR's and the
x87's exception flags are not updated and unmasked floating-point exceptions
are not raised.

`x64_run` returns at a stop address, after an instruction budget, or when the
guest calls or jumps to an address the host says is native code; the host then
runs that function and `x64_return_from_native` resumes the guest.
`x64_state_from_context`/`x64_state_to_context` convert to and from Wine's
AMD64 `CONTEXT`, and `x64_fxsave`/`x64_fxrstor` to and from the FXSAVE image
32-bit contexts keep their x87, MMX and SSE state in.

The interpreter needs nothing from the host but memory functions and `sin`,
`cos`, `tan`, `atan2`, `log`, `pow` (which ntdll exports too): it sets the
rounding mode through MXCSR or FPCR itself, has exact `trunc`, `nearbyint`,
`fmod` and `remainder` of its own, divides 128 by 64 bits without the compiler
runtime, and reads time through `NtQueryPerformanceCounter` in a Windows build.
That is what lets it run inside a WoW64 CPU module.

Interpreting is much slower than running translated code. On the x86-64 Linux
machine used for development (an Intel Xeon at 2.3 GHz) it runs 40 to 70
million x86 instructions a second; the engine check reports the rate on the
device. A cache of decoded instructions, threaded dispatch and flags computed
only when read would make it several times faster; it will still be much
slower than JIT, so No JIT suits older, lighter games.

## The WoW64 CPU module

`build/wow64-interp/cpu.c` is a CPU module for Wine's `wow64.dll` that runs
32-bit code with the interpreter (`build.sh` builds `wow64interp.dll` for
x86-64 with mingw-w64 GCC and for ARM64 with clang and lld). It imports only
`ntdll` and `wow64`, as a module on the 64-bit side of a WoW64 process must,
and exports the `BTCpu` entry points `wow64.dll` looks up, as Wine's
`wow64cpu.dll` and FEX's `xtajit.dll` do.

- The 32-bit registers live in the thread's WoW64 CPU area whenever 32-bit code
  is not running, as with `wow64cpu.dll`, so `wow64.dll` and ntdll read and
  change them there; a context they replace (APCs, exceptions, `NtContinue`)
  is marked and reloaded.
- `wow64.dll` points i386 ntdll's system call and unix call dispatchers at two
  addresses below 4 GB that the module hands out. Nothing executes them: the
  interpreter stops when a `CALL` or `JMP` reaches either, and the module calls
  `Wow64SystemServiceEx` or `__wine_unix_call_dispatcher` on the 64-bit side.
- Exceptions the interpreter finds (`INT3`, `INT 29h` fast fail, divide errors,
  illegal and privileged instructions) are raised on the 32-bit side through
  `Wow64PassExceptionToGuest`. An access to a bad address faults inside the
  interpreter, in 64-bit code: ntdll's dispatcher calls
  `BTCpuResetToConsistentState`, which puts the 32-bit state back at the
  faulting instruction, and `wow64.dll`'s handler passes the exception on.
  Guard pages of the 32-bit stack are grown by Wine as for any access.
- The guest window base B comes from
  `NtQueryInformationProcess(ProcessWineIosWowGuestBase)`, as for Madeira's
  other WoW64 code; on upstream Wine the query fails and B is 0.

Not yet: it has not run in Wine (step 2 above). Another thread's
`GetThreadContext` sees the registers as of that thread's last system call.

## Testing

`python3 tests/host/check-x64-interpreter.py` (also run in CI, on the macOS
runner) does what the host it runs on allows:

- **Everywhere:** builds the interpreter and the engine check with warnings as
  errors and runs the engine check. On an ARM64 Mac that is the iPhone's
  architecture.
- **x86-64 Linux:** runs the committed programs natively (the 32-bit ones in a
  32-bit process with Windows' x87 control word) to confirm the expected
  results, then two instruction fuzzers. `gen-fuzz-insns.py` writes about
  1,400 64-bit snippets (every instruction form above, in each operand size,
  with register, memory, `LOCK` and segment forms, branches taken and not);
  `fuzz.c` runs each natively and through the interpreter, at the same
  address, from the same random registers, flags, XMM registers, MXCSR
  (rounding, DAZ, FTZ) and memory, and compares every register, the defined
  flags, MXCSR and both memory regions. `gen-fuzz-insns32.py` writes about 800
  32-bit snippets (integer code as old compilers emit it, x87, MMX); `fuzz32.c`
  runs them natively in a 32-bit process from random x87, MMX and SSE state
  (`FXRSTOR`), and `fuzz replay` compares the interpreter's registers, flags,
  x87 stack, status and tag words, MMX, XMM and memory.
- **With `qemu-aarch64` and an `aarch64-linux-gnu` toolchain:** replays the
  recorded native results through an ARM64 build, and runs the engine check on
  ARM64.
- **With mingw-w64 GCC, or clang with lld:** builds the WoW64 CPU module and
  checks what it imports and exports.

During development the 64-bit fuzzer ran 3.3 million comparisons over four
seeds and the 32-bit one 1.19 million over five, with no difference, on
x86-64 and replayed on ARM64. Deliberate bugs (a wrong flag, a NaN rule, a
conversion result, the x87 `DC` operand order, `FIST` rounding, `FCOM`'s
condition codes, an MMX unpack, `PUSHA`'s ESP and others) were each caught.

`tests/x64-interp/gen-selfcheck.py` rebuilds the programs and
`app/Madeira/NoJITSelfCheckData.h` (it runs them natively, so it needs x86-64
Linux with GCC, its 32-bit support and binutils).
