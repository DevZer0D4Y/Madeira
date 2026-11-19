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

1. **The CPU module** (done): `build/wow64-interp`, below.
2. **Run it in Wine on a PC** (done): in Wine 11.4 built for x86-64 Linux,
   32-bit programs run through it and print what they print on the processor
   (Testing, below). Madeira's own Wine does not build for Linux (its ntdll
   uses Apple's APIs), so this was upstream Wine 11.4, the release Madeira's
   is based on.
3. **Run it in Madeira with JIT on** (ready to try): put `wow-cpu = interpreter`
   in `Documents/madeira.cfg`, or in a game's own lines, or choose it in
   Settings › Memory & sync › All settings, under x86 emulation (FEX):
   "32-bit programs run on: Madeira's x86 interpreter (testing)". The
   32-bit programs of that session then run on the interpreter instead of FEX,
   with JIT on as before (the native modules still need it). It is much slower
   than FEX: try older, lighter 32-bit games. The log shows
   `[WineProc] wow-cpu = interpreter: ...` when the session uses it, then
   `wow64interp: this 32-bit process runs on Madeira's x86 interpreter` for each
   32-bit process, and `wow64interp: unsupported instruction at ...` with the
   instruction's bytes if a program uses one the interpreter does not have.
   What needs the device: Madeira's guest window (B is not 0 there, and the
   module follows its rules as FEX's module does, but on Linux B is 0) and
   the speed of the iPhone's cores.
4. **The native modules as signed code**: convert them to Mach-O when
   Madeira is built, so the sideloader signs them with the app and iOS runs
   them like any other library. The open upstream pull request #497 (another
   contributor's) does this without changing their code: `build/signed-pe/
   prepare.py` maps a PE's sections into a Mach-O image, its data relocations
   becoming dyld rebases, and Wine's loader runs a prepared image's code from
   the signed copy (`MADEIRA_SIGNED_PE_DIR`). It is built for ARM64EC, and
   has run Wine's `ntdll` that way on an iPhone with JIT off. For the 32-bit
   path, whose four modules are plain ARM64, checked against its rules:
   - `wow64.dll`, `wow64win.dll` and `wow64interp.dll` already have the layout
     it needs (64 KiB sections, no relocations in code, no pointers outside the
     image; `build.sh` links `wow64interp.dll` as winegcc links Wine's modules,
     and `check-x64-interpreter.py` checks it). `ntdll.dll` has 258 relocations
     in its code, all the same thing: each system call stub's literal holding
     `__wine_syscall_dispatcher`'s address (Wine's ARM64 `__ASM_SYSCALL_FUNC`
     in `include/wine/asm.h`).
   - All four read the TEB through x18: 661, 49, 38 and 10 times. The pull
     request's offline TEB rewriting uses x13 and x14, which ARM64EC reserves
     and plain ARM64 does not.

   `build/signed-arm64/rewrite.py` (done) prepares them, without changing
   Wine's source:
   - Each system call stub loads the dispatcher's address PC-relatively
     (`adrp`/`ldr`, as Wine's ARM64EC stub does), its literal becomes zeros and
     its relocation entry is cleared. The pull request's `read_pe` refuses the
     original `ntdll.dll` ("Native text relocations ... need an offline code
     rewrite") and accepts the rewritten one.
   - Each instruction that uses x18 becomes a branch to a trampoline that saves
     two free registers on the stack (plain ARM64 keeps SP 16-byte aligned),
     reads the TEB from the TSD slot the way the loader's `ios_jit_patch_x18`
     does (`TPIDRRO_EL0 & ~7`, plus the slot's offset), runs the instruction
     with that register in place of x18, restores the two and branches back.
     The offset is read from a data slot the loader fills, the one the pull
     request uses (offset 4 of `madeira_pe_thread_state`), not built into the
     code. That covers all 758 sites, including ntdll's x18 store in the
     routine that captures a context, which `ios_jit_patch_x18` leaves alone.
     It refuses what it cannot rewrite (an instruction that writes x18, or
     uses SP beside it); none of the four modules has one.
   - In the pull request's `prepare.py`, plain ARM64 needs these lines where
     it now refuses anything but ARM64EC (in `thread_state.rewrite`, or before
     calling it), after `rewrite.py IN OUT` has done the stubs on the file:
     ```python
     if machine == 0xaa64:
         trampolines, teb_sites = rewrite_image(image, PE_OFFSET, tramp_start, slot)
         return bytearray(trampolines), teb_sites, 0
     ```
     With that, `prepare.py` builds the Mach-O framework of each of the four
     modules (tried on Linux, code signing left out): the PE's code in
     `__TEXT`, its 758 trampolines in `__THREAD`, every branch reaching its
     trampoline, every trampoline reading the slot and branching back.

   What was not tried: running them. That needs an iPhone, the pull
   request's loader (`signed_pe_ios.c`) used for these four modules, and the
   loader writing the TSD offset to the slot. As with the load-time patcher,
   the trampolines have no unwind data. That matters only if an instruction in
   one faults: they read or write the TEB, except ntdll's store of the TEB into
   a context its caller passed.
   - A signed library has one copy of its data, and every Windows process in
     Madeira is a pseudo-process in the same iOS process: two processes that
     load `ntdll.dll` need two copies, and the pull request's loader refuses a
     second one. A 32-bit game's process is one; the 64-bit helpers a session
     may start (`services.exe`, `rpcss.exe`, `explorer.exe`) are more, and
     their own DLLs are native code too. Signing a few numbered copies of each
     module, and starting no helper a game does not need, would cover a
     session.
5. **A 32-bit session that starts without the JIT pool**: skip
   `StikJITHelper`'s pool for No JIT, and have the loader use the signed copies.
6. **Speed**, below.

64-bit games need more: every one of Wine's ARM64EC modules as signed code
(416 files today; pull request #497 works toward that, with FEX translating
programs ahead of time), and for the interpreter, a version of FEX's ARM64EC
module that interprets (`libarm64ecfex.dll`, Wine's `BTCpu64` interface).

## What the option does now

- Madeira asks for no debugger. **Play** says that No JIT cannot start games
  yet instead of enabling JIT, and **Enable JIT** says that No JIT is selected.
  Choosing Automatic, StikDebug or Built-in StikJIT plays as before.
- **JIT setup** (with No JIT selected) has **Run engine check**. It runs 25
  small x86 programs, each with two inputs, through the interpreter from
  ordinary heap memory that is never executable, and compares each of the 50
  results with the one a real x86 CPU gave for the same bytes, running each
  program twice: with the decoded-instruction cache, as games run, and
  without it. It reports the instructions run, the rate both ways, and whether
  JIT was off (`CS_DEBUGGED` clear) during the check. The log gets a `[no-jit]` line with the same result.

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
carries it out in C. Nothing is generated: there is no executable memory at
all. All state is in one `X64State` per thread.

Each thread can have a **decoded-instruction cache** (`X64Env.cache`, 512 KB)
of **blocks**: straight runs of up to 14 decoded instructions, at most 48 bytes
of one page, ending after a jump, call or return. The common forms (moves,
`LEA`, the ALU operations, `TEST`, `INC`/`DEC`, `PUSH`/`POP`, jumps, calls and
returns, `MOVZX`/`MOVSX`, `SETcc`, `CMOVcc`, `IMUL`, `MUL`/`DIV`, shifts and
double shifts, `CDQ`) run from the decoded form, and x87, SSE and MMX
instructions run from it through the same code the decoder uses for them; a
block ends before anything else (string instructions, `LOCK`, system
instructions...), which runs through the decoder as before. The most common
of all (32- and 64-bit register `MOV`, `ADD`, `SUB`, `CMP`, `AND`, `OR`, `XOR`,
`TEST`, shifts by a constant, and loads and stores) are decoded into forms of
their own, and the loop that runs blocks is built once for 32-bit and once for
64-bit code, so it does not test the mode each time. Each time a block starts,
its bytes are compared with the ones it was decoded from, and it is decoded
again if they changed; an instruction that stores within 128 bytes of its own
block's code ends the block there, so it is compared again before the rest
runs. Self-modifying code and code a game's own JIT writes (Mono, .NET, LuaJIT)
need nothing special. Each address has two places in the cache, so two blocks
that run in turn do not keep replacing each other. Arithmetic flags are worked
out lazily: `ADD`, `SUB`, `CMP`, `ADC`, `SBB`, the logic operations, `INC`,
`DEC` and `SHL`/`SHR`/`SAR` leave their operands, and the flags are computed only
when something reads them (a `CMP` followed by a conditional jump reads just the
condition).

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
  stores, moves, integer and BCD conversions and comparisons. `FADD`, `FSUB`,
  `FMUL`, `FDIV` and `FSQRT` give the x87's results bit for bit at each
  precision the control word can ask for: 53 bits (Windows' default, 0x27F),
  24 (what Direct3D 9 sets unless a game asks it not to) and 64 (Delphi's
  default, and MinGW's), in each rounding direction, with x87's exponent
  range, tiny and huge results, NaN rules, and the default NaN for invalid
  operands (unnormals, pseudo-NaNs, an empty register). Values a double holds
  take the host's double arithmetic at 24 and 53 bits; the rest, and all of
  64-bit precision, take integer arithmetic on the 64-bit significands.
  `FSCALE`, `FRNDINT`, `FXTRACT`, conversions and comparisons are exact too
  (`FST m32` and `FST m64` round once, to denormals as well).
  Transcendental results (`FSIN`, `FPTAN`, `FYL2X`, `F2XM1`...) are double's,
  which can differ from the x87's in the last bit or two (the x87 does not
  round these correctly either, and Intel's and AMD's differ); far from zero,
  `FSIN` and the like reduce their argument exactly, where the x87 uses a
  66-bit pi;
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
machine used for development (an Intel Xeon at 2.3 GHz, shared, so timings vary
by a third from run to run) the engine check runs about 110 to 150 million x86
instructions a second with the cache (35 without); the engine check reports both
rates on the device. Blocks halved the work: the cached path went from about
170 host instructions per x86 instruction (one cached instruction at a time) to
about 75 (counted with callgrind, which does not vary), and from about 85 to 130
million a second on the same machine, run after run. Under QEMU on ARM64 the
engine check went from 9.4 to 14 million a second. It will still be much slower
than JIT, so No JIT suits older, lighter games.

## The WoW64 CPU module

`build/wow64-interp/cpu.c` is a CPU module for Wine's `wow64.dll` that runs
32-bit code with the interpreter (`build.sh` builds `wow64interp.dll` for
x86-64 with mingw-w64 GCC, and for ARM64 with llvm-mingw, or clang and lld;
`build.sh --install` puts the ARM64 one in `app/Madeira/aarch64-windows/`,
which `build/windows-runtime/build.sh` runs beside FEX's module). It imports only
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

- In Madeira, `wow-cpu = interpreter` makes the session's `system32\xtajit.dll`
  (and `sysaa64\xtajit.dll`) link to the farm's `wow64interp.dll`
  (`app/Madeira/WineProcessBridge.m`), which is the file `wow64.dll` loads; the
  loader keeps the module in its JIT-pool copy only, as it does FEX's.
- On ARM64 it reads the TEB (x18) with one `MOV`: Madeira's loader rewrites x18
  reads on iOS, where x18 is not the TEB, in a few instruction forms only, and
  `check-x64-interpreter.py` fails if the module has any other.
- It is built for the A12 (`-mcpu=apple-a12`), the oldest chip iOS 17 runs on: the
  guest's loads are `LDAPR` (ARMv8.3), which keeps x86's memory order without
  `LDAR`'s wait for earlier stores (x86 lets a load pass an earlier store too), and
  `LOCK`ed instructions use ARMv8.1 atomics.
- Its messages go to Wine's debug output (`__wine_dbg_output`), which Madeira's
  log shows whatever `WINEDEBUG` says.
- On an x86-64 host, Wine keeps a 32-bit thread's x87, MMX and SSE registers
  in the processor rather than the CPU area (that is where `wow64cpu.dll`
  leaves them), so the module puts the program's there while 64-bit code runs
  for it and takes them back after. On ARM64, as in Madeira, they are in the
  CPU area like the rest.

Another thread's `GetThreadContext` sees the registers as of that thread's
last system call: Wine does not ask the CPU module when it suspends a thread,
and FEX's module in Madeira has the same limit today.

## Testing

`python3 tests/host/check-x64-interpreter.py` (also run in CI, on the macOS
runner) does what the host it runs on allows:

- **Everywhere:** builds the interpreter and the engine check with warnings as
  errors and runs the engine check. On an ARM64 Mac that is the iPhone's
  architecture.
- **x86-64 Linux:** runs the committed programs natively (the 32-bit ones in a
  32-bit process with Windows' x87 control word) to confirm the expected
  results, then two instruction fuzzers. `gen-fuzz-insns.py` writes about
  1,900 64-bit snippets (every instruction form above, in each operand size,
  with register, memory, `LOCK` and segment forms, branches taken and not);
  `fuzz.c` runs each natively and through the interpreter, at the same
  address, from the same random registers, flags, XMM registers, MXCSR
  (rounding, DAZ, FTZ) and memory, and compares every register, the defined
  flags, MXCSR and both memory regions. `gen-fuzz-insns32.py` writes about 1,350
  32-bit snippets (integer code as old compilers emit it, x87, MMX); `fuzz32.c`
  runs them natively in a 32-bit process from random x87, MMX and SSE state
  (`FXRSTOR`), and `fuzz replay` compares the interpreter's registers, flags,
  x87 stack, status and tag words, MMX, XMM and memory.
- **With `qemu-aarch64` and an `aarch64-linux-gnu` toolchain:** replays the
  recorded native results through an ARM64 build, and runs the engine check on
  ARM64.
- **With mingw-w64 GCC, or clang with lld:** builds the WoW64 CPU module and
  checks what it imports and exports.

`python3 tests/host/check-arm64-teb-rewrite.py` (not in CI: it needs LLVM's
`llvm-mc`) checks `build/signed-arm64/rewrite.py` (step 4) on the four modules
in `app/Madeira/aarch64-windows`, with llvm-mc as the decoder rather than the
rewriter's own:
- every instruction that uses x18 or w18, outside literal data, is a branch to
  a trampoline, and nothing else is; each trampoline is the expected sequence,
  with the original instruction renamed to a register it does not use, and
  branches back to the next instruction;
- each system call stub loads the same address PC-relatively, no relocation is
  left in code, and nothing outside the code changed but cleared relocation
  entries;
- done as a wrapper would (the stubs on the file, then `rewrite_image` on the
  image mapped elsewhere), the code and trampolines are the same;
- with `qemu-aarch64` and an `aarch64-linux-gnu` toolchain, every distinct x18
  instruction in the modules (133) and about 4,700 random encodings of every
  form the rewriter knows run as the original (x18 = a TEB) and as rewritten
  (x18 garbage, the TEB found through `TPIDR_EL0`, which Linux lets a program
  set, and the slot), from the same registers, flags and memory
  (`tests/x64-interp/teb-runner.c`): registers, flags and memory match after,
  and x18 is untouched. Deliberate bugs (no `& ~7`, a scratch register the
  instruction uses, a register not restored, a missed register field or
  instruction class, a stub's offset, CMP refused) were each caught.

`python3 tests/host/check-wow64interp-wine.py WINE_BUILD_DIR` (not in CI:
it needs a Wine built for x86-64 Linux with `--enable-archs=i386,x86_64`,
about an hour's build) installs `wow64interp.dll` in a new prefix and runs two
32-bit programs with `wow64cpu.dll`, on the processor, and with
`wow64interp.dll`, and compares what they print. `wowtest.c` raises
exceptions from the processor (access violations, a divide error, `INT3`,
`UD2`, `INTO`, a stack overflow) and resumes from a changed context, checks
the registers its handler sees, and runs threads with atomics, x87 math through
msvcrt, 64-bit division, files and sockets (system and unix calls);
`workloads-main.c` runs the engine check's 32-bit programs. Every line is the
same, except that msvcrt's transcendental results are compared to 1e-13. A
simple loop there runs about 15 times slower interpreted than on the
processor: about 190 million instructions a second (twice what it ran before
the cache held blocks). The module gives each
thread its own cache and frees it when the thread ends.

During development the 64-bit fuzzer ran 3.3 million comparisons over four
seeds and the 32-bit one 1.19 million over five, with no difference, on
x86-64 and replayed on ARM64. Deliberate bugs (a wrong flag, a NaN rule, a
conversion result, the x87 `DC` operand order, `FIST` rounding, `FCOM`'s
condition codes, an MMX unpack, `PUSHA`'s ESP and others) were each caught.
The 32-bit fuzzer's x87 arithmetic snippets also run from any 80-bit values
(full significands, every exponent, denormals, NaNs, unsupported encodings)
at random precision and rounding control: 2 million comparisons over six seeds
with no difference, and eleven deliberate bugs in the exact arithmetic (ties to
even, a sticky bit, denormals, overflow, a square root's last bit, the sign of
an exact zero...) each caught.

Every other fuzz run goes through the decoded-instruction cache, which keeps
its entries from run to run, so both the cached path and the decoder alone are
compared with the processor, and so are cache hits and code that changes under
the cache. Snippets chain each kind of flag-setting instruction into each
condition, `ADC`, `INC`, `PUSHF` and shifts (lazy flags carry across
instructions), and two modify their own code as they run. Deliberate bugs in
the cache and the lazy flags (a signed condition, `INC` keeping `CF`, a byte
register, `MOVSX`, a segment base, the bytes check, a shift reading stale flags,
`ADC`'s and `SBB`'s carry in and out...) were each caught.

For the blocks: snippets store into an instruction a few after the store in their
own block, which the interpreter has already decoded, with each kind of store (32-
and 64-bit, read-modify-write, SSE, x87); others use the register forms that
compilers such as MSVC emit and assemblers do not (`03 C3`, `ADD EAX, EBX`); shifts by
a constant are followed by each flag reader; and every third run goes in pieces of
one to five instructions (`X64_EXIT_BUDGET` between them), each of which must run
exactly its budget, so blocks stop and resume part of the way through. Deliberate
bugs (no store check, either kind; a lazy shift flag; the shift's carry into `ADC`;
the `03 C3` operands; `SAR` as a logical shift; a block moved to its second place
without its instructions; the bytes check missing the last bytes; the stop address
or the budget inside a block; a 32-bit load not clearing the upper half) were each
caught: the second place by the engine check, whose programs are big enough to
need it, the rest by the fuzzers. Before these snippets were added, four of them
were not.

`tests/x64-interp/gen-selfcheck.py` rebuilds the programs and
`app/Madeira/NoJITSelfCheckData.h` (it runs them natively, so it needs x86-64
Linux with GCC, its 32-bit support and binutils).
