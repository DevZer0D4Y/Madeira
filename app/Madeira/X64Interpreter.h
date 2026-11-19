// SPDX-License-Identifier: GPL-3.0-or-later
// Madeira Converter Exception: see LICENSE-EXCEPTION.md
//
// An x86-64 interpreter that runs guest code without executable memory.
//
// FEX translates x86-64 code to ARM64 at run time, which needs memory that is
// both written and executed: on iOS that means JIT, and JIT means a debugger.
// This interpreter instead reads each instruction as data and carries it out
// in C, so it runs in an ordinary signed process with no debugger attached.
// It is much slower than translated code and is the first part of the
// No JIT mode (docs/NO-JIT.md); nothing in the game path uses it yet.
//
// Guest memory is host memory: guest addresses are host pointers, as they are
// under FEX in Wine's ARM64EC process. The interpreter is reentrant: all of its
// state is in X64State, one per guest thread.

#ifndef X64_INTERPRETER_H
#define X64_INTERPRETER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    X64_RAX, X64_RCX, X64_RDX, X64_RBX, X64_RSP, X64_RBP, X64_RSI, X64_RDI,
    X64_R8, X64_R9, X64_R10, X64_R11, X64_R12, X64_R13, X64_R14, X64_R15,
};

typedef union X64Vec {
    uint8_t b[16];
    uint16_t w[8];
    uint32_t d[4];
    uint64_t q[2];
} X64Vec;

/// An x87 register as x86 stores it: a 64-bit significand with its explicit
/// integer bit, then sign and 15-bit exponent (the layout of an FXSAVE slot).
typedef struct X87Reg {
    uint64_t mantissa;
    uint16_t exponent;      // sign in bit 15
    uint16_t pad[3];
} X87Reg;

typedef struct X64State {
    uint64_t gpr[16];           // RAX RCX RDX RBX RSP RBP RSI RDI R8-R15
    uint64_t rip;
    uint64_t rflags;
    uint64_t fs_base, gs_base;  // GS is the TEB on Windows
    X64Vec xmm[16];
    uint32_t mxcsr;
    uint16_t fcw, fsw;          // x87 control and status words; TOP is fsw bits 11-13
    uint16_t seg_cs, seg_ds, seg_es, seg_fs, seg_gs, seg_ss;
    uint64_t instructions;      // instructions retired, for statistics
    // x87 registers by physical number: ST(i) is st[(TOP + i) & 7]. The MMX
    // registers are their significands.
    X87Reg st[8];
    uint8_t ftw;                // abridged tag word: bit i set while st[i] holds a value
    // 32-bit (WoW64) code: 32-bit registers, addresses and stack. Guest address a
    // is host address guest_base + a (Madeira's guest window; 0 elsewhere).
    bool mode32;
    uint64_t guest_base;
    uint64_t insn_start;        // the instruction being executed, for fault reporting
    // Arithmetic flags not yet folded into rflags: the last ADD, SUB, CMP, logic
    // operation, INC or DEC leaves its operands here, and the flags are worked out
    // when something reads them. x64_run folds them in before it returns.
    uint64_t lazy_a, lazy_b, lazy_r;
    uint8_t lazy_op, lazy_size, lazy_carry;
} X64State;

typedef enum X64Exit {
    X64_EXIT_STOP,          // rip reached X64Env.stop_rip
    X64_EXIT_BUDGET,        // the instruction budget ran out at an instruction boundary
    X64_EXIT_NATIVE,        // a CALL or JMP went to an address is_native accepted: rip is the
                            // target, and a CALL's return address is on the guest stack
    X64_EXIT_SYSCALL,       // SYSCALL: rip is after it
    X64_EXIT_BREAKPOINT,    // INT3 or INT1: rip is after it, as x86 reports a breakpoint
    X64_EXIT_INTERRUPT,     // INT n (n in X64ExitInfo.vector, 0x29 is __fastfail): rip is after it
    X64_EXIT_DIVIDE_ERROR,  // DIV or IDIV by zero or with a quotient too large: rip is at it
    X64_EXIT_ILLEGAL,       // an instruction x86-64 does not have (UD2 included): rip is at it
    X64_EXIT_PRIVILEGED,    // HLT, CLI, STI and other kernel-mode instructions: rip is at it
    X64_EXIT_UNSUPPORTED,   // a valid instruction this interpreter does not implement: rip is at it
} X64Exit;

typedef struct X64Env {
    // The run stops before executing the instruction at this address. A host that
    // calls into guest code pushes it as the return address.
    uint64_t stop_rip;
    // Whether a CALL or JMP target is native (host) code. NULL: none is.
    bool (*is_native)(void *user, uint64_t target);
    void *user;
    // Optional: X64_CACHE_SIZE bytes of zeroed memory, aligned to 64, that one
    // thread at a time runs with (NULL: none). x64_run keeps blocks of the
    // instructions it decodes there, and runs the common ones from them. Each time
    // it runs a cached block it checks that the block's bytes are still the same,
    // so code that changes (self-modifying code, a game's own JIT) is decoded again.
    void *cache;
} X64Env;

#define X64_CACHE_BLOCKS_LOG2 10
#define X64_CACHE_BLOCK_SIZE 512u
#define X64_CACHE_SIZE ((1u << X64_CACHE_BLOCKS_LOG2) * X64_CACHE_BLOCK_SIZE)

typedef struct X64ExitInfo {
    uint64_t rip;           // address of the instruction that stopped the run
    uint8_t bytes[16];      // its first bytes (for X64_EXIT_UNSUPPORTED and friends)
    uint8_t length;         // how many of them were decoded
    uint8_t vector;         // INT n
} X64ExitInfo;

/// Registers as a new Windows thread has them: flags 0x202, MXCSR 0x1F80,
/// x87 control word 0x27F, user-mode selectors. 64-bit mode.
void x64_state_init(X64State *state);

/// The same for a 32-bit (WoW64) thread whose guest address 0 is host address
/// `guest_base`.
void x64_state_init32(X64State *state, uint64_t guest_base);

/// Runs guest code from state->rip until an exit. `budget` is the most
/// instructions to run (UINT64_MAX: no limit); a REP string instruction counts
/// once. The host's rounding mode follows MXCSR while it runs and is restored
/// before it returns. `info` may be NULL.
X64Exit x64_run(X64State *state, const X64Env *env, uint64_t budget, X64ExitInfo *info);

/// Folds pending arithmetic flags into state->rflags. For a host that reads the
/// state while x64_run is still running it (a fault handler); after x64_run
/// returns, rflags is already complete.
void x64_flush_flags(X64State *state);

/// Finishes a call that left with X64_EXIT_NATIVE: RAX takes the native
/// function's result and the return address is popped into rip. In 32-bit mode
/// EAX takes the low half and EDX the high half (a 64-bit cdecl result).
void x64_return_from_native(X64State *state, uint64_t rax);

/// The 512-byte FXSAVE image of the x87, MMX, SSE and MXCSR state, as Windows
/// keeps it in CONTEXT.FltSave and I386_CONTEXT.ExtendedRegisters. Bytes 416-511
/// are left as they are.
void x64_fxsave(const X64State *state, void *area);
void x64_fxrstor(X64State *state, const void *area);

/// The CPUID answers the interpreter gives: a fixed profile of what it implements
/// (x87, MMX, SSE, SSE2, SSE3, CMPXCHG16B, POPCNT, LZCNT; no AVX).
void x64_cpuid(uint32_t leaf, uint32_t subleaf, uint32_t out[4]);

const char *x64_exit_name(X64Exit exit);

/// Wine's AMD64 CONTEXT (also the layout of ARM64EC_NT_CONTEXT): 1232 bytes.
#define X64_CONTEXT_SIZE 1232u
void x64_state_from_context(X64State *state, const void *context);
void x64_state_to_context(const X64State *state, void *context);

#ifdef __cplusplus
}
#endif

#endif // X64_INTERPRETER_H
