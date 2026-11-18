# No JIT (experimental)

**Settings → JIT → No JIT (experimental)** is the start of a way to run
Windows code with no debugger attached: no StikDebug, no pairing file, no
LocalDevVPN. It does not start games yet. This page says what it does today,
how the part that exists is tested, and what is still missing.

## Why Madeira needs JIT

iOS lets a process write code and then run it only while a debugger is
attached (`CS_DEBUGGED`); [JIT setup](JIT.md) is how Madeira gets one. Two
things in a game session need that:

1. **The game's x86 code.** FEX translates it to ARM64 as it runs, writing the
   translation into memory and then executing it.
2. **Wine's own Windows libraries** (`ntdll.dll`, `kernel32.dll`, DXMT,
   madeira-d3d12 and the rest). They are ARM64EC PE files, not Mach-O, so iOS
   cannot load them as signed code: Madeira copies their code into the JIT pool
   and runs it from there.

No JIT replaces the first with an interpreter. The second still needs JIT,
which is why games cannot start in this mode yet.

## What the option does now

- Madeira asks for no debugger. **Play** says that No JIT cannot start games
  yet instead of enabling JIT, and **Enable JIT** says that No JIT is selected.
  Choosing Automatic, StikDebug or Built-in StikJIT plays as before.
- **JIT setup** (with No JIT selected) has **Run engine check**. It runs 12
  small x86-64 programs, each with two inputs, through the interpreter from
  ordinary heap memory that is never executable, and compares each of the 24
  results with the one a real x86-64 CPU gave for the same bytes. It reports the instructions run, the rate, and
  whether JIT was off (`CS_DEBUGGED` clear) during the check. The log gets a
  `[no-jit]` line with the same result.

The programs are in `tests/x64-interp/workloads.c`: CRC-32, quicksort, a prime
sieve, 8- to 128-bit division and multiplication, double and float matrix
work, SSE2/SSE3 vector code, `REP` string instructions (forward, backward and
overlapping), jump tables and function pointers, `LOCK`ed atomics including
`CMPXCHG16B`, every MXCSR rounding mode with flush-to-zero, and calls from x86
code into host code and back. They are compiled with the Windows x64 calling
convention, as games are.

## The interpreter

`app/Madeira/X64Interpreter.c` decodes each x86-64 instruction from guest
memory and carries it out in C. Nothing is generated or cached, so
self-modifying code and code a game's own JIT writes (Mono, .NET, LuaJIT) need
nothing special. Guest addresses are host addresses, as they are under FEX in
Wine's ARM64EC process, and all state is in one `X64State` per thread.

Implemented:

- the whole general-purpose instruction set, with x86's flags; `LOCK` prefixes
  as real host atomics, and `XCHG` with memory always atomic;
- string instructions with `REP`, `REPE` and `REPNE`, in both directions;
- SSE, SSE2, SSE3, `PSHUFB` and `PALIGNR`;
- `CMPXCHG8B`/`CMPXCHG16B`, `POPCNT`, `LZCNT`, `TZCNT`, `BSWAP`, `CPUID`,
  `RDTSC`, fences, `LDMXCSR`/`STMXCSR`, `FXSAVE`/`FXRSTOR`, and the x87
  control-word instructions C runtimes run at start-up;
- FS and GS segment bases (GS is the TEB on Windows).

Floating point follows x86 rather than the host: which NaN an operation
returns and the default NaN, MXCSR rounding, DAZ and FTZ, and the "integer
indefinite" result of an out-of-range conversion. On ARM64, aligned loads are
acquire and aligned stores release, which keeps x86's ordering of memory
between threads.

`CPUID` describes what is implemented (SSE3, `CMPXCHG16B`, `POPCNT`, `LZCNT`;
no AVX), so a program that checks before using newer instructions picks a path
the interpreter has.

Not implemented: x87 arithmetic, MMX, AVX and anything VEX- or EVEX-encoded,
SSSE3 beyond `PSHUFB`/`PALIGNR`, SSE4, 32-bit (WoW64) code, and system
instructions. Any of these stops the run with `X64_EXIT_UNSUPPORTED` (or
`X64_EXIT_PRIVILEGED`) before the instruction changes anything, with the
instruction's address and bytes, so the caller can report it. MXCSR's exception
flags are not updated and unmasked floating-point exceptions are not raised.

`x64_run` returns at a stop address, after an instruction budget, or when the
guest calls or jumps to an address the host says is native code; the host then
runs that function and `x64_return_from_native` resumes the guest. That is how
interpreted game code will call Wine. `x64_state_from_context` and
`x64_state_to_context` convert to and from Wine's 1,232-byte AMD64 `CONTEXT`.

Interpreting is much slower than running translated code. On the x86-64 Linux
machine used for development (an Intel Xeon at 2.3 GHz) it runs 50 to 80
million x86 instructions a second; the engine check reports the rate on the
device.

## Testing

`python3 tests/host/check-x64-interpreter.py` (also run in CI, on the macOS
runner) does what the host it runs on allows:

- **Everywhere:** builds the interpreter and the engine check with warnings as
  errors and runs the engine check. On an ARM64 Mac that is the iPhone's
  architecture.
- **x86-64 Linux:** runs the committed programs natively to confirm the
  expected results, then the instruction fuzzer. `gen-fuzz-insns.py` writes
  about 1,400 snippets (every instruction form above, in each operand size,
  with register, memory, `LOCK` and segment forms, branches taken and not).
  `fuzz.c` runs each one natively and through the interpreter, at the same
  address, from the same random registers, flags, XMM registers, MXCSR
  (rounding, DAZ, FTZ) and memory, and compares every register, the defined
  flags, MXCSR and both memory regions.
- **x86-64 Linux with `qemu-aarch64` and an `aarch64-linux-gnu` toolchain:**
  records the native fuzz results and replays them through an ARM64 build of
  the interpreter, and runs the engine check on ARM64.

During development the fuzzer ran 3.3 million native-against-interpreted
comparisons over four seeds with no difference, and 166,680 recorded results
replayed on ARM64 with none. Ten deliberate bugs (a wrong flag, a NaN rule, a
conversion result, a sign extension, the direction flag in a string copy and
others) were each caught.

`tests/x64-interp/gen-selfcheck.py` rebuilds the programs and
`app/Madeira/NoJITSelfCheckData.h` (it runs them natively, so it needs x86-64
Linux with GCC and binutils).

## Before a game can start

1. **Wine's libraries as signed code.** Convert the ARM64EC PE modules to
   Mach-O libraries when Madeira is built, so the sideloader signs them with
   the app and iOS loads them like any other library. The open upstream pull
   request #497 does this for `ntdll`, `kernelbase` and `kernel32`, and runs
   test programs on an iPhone with JIT off.
2. **An emulator module for Wine backed by the interpreter.** Wine's ARM64EC
   loader drives x86-64 code through an emulator DLL (FEX's
   `libarm64ecfex.dll`): process and thread start-up, running code, notices
   that code memory changed, and exception contexts. A version of it that calls
   `x64_run` instead of FEX's JIT replaces the first use of the JIT pool.
3. **A session that starts without the JIT pool.** Starting Wine reserves and
   maps the pool through the debugger (`StikJITHelper`); No JIT needs that step
   skipped and the loader's pool copies turned off.
4. **x87, AVX and 32-bit code** for games that use them: 32-bit games use the
   x87 instructions throughout, and many recent games need AVX.
5. **Speed.** A cache of decoded instructions, threaded dispatch and flags
   computed only when read would make the interpreter several times faster.
   It will still be much slower than JIT, so No JIT suits lighter games.
