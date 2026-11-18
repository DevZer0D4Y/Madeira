// SPDX-License-Identifier: GPL-3.0-or-later
// Madeira Converter Exception: see LICENSE-EXCEPTION.md
//
// What fuzz.c (64-bit: runs, records and replays) and fuzz32.c (a 32-bit
// process: records) share: the fixed memory layout, the snippet table, and the
// random inputs, which both must generate identically from the same seed.

#ifndef FUZZ_COMMON_H
#define FUZZ_COMMON_H

#include "X64Interpreter.h"

#include <math.h>
#include <setjmp.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>

#define CODE    0x30000000ul
#define ANCHOR  0x30100000ul
#define MEM     0x30200000ul
#define MEMSZ   0x4000ul
#define STACK   0x30300000ul
#define STACKSZ 0x2000ul

// Template flags, as gen-fuzz-insns.py and gen-fuzz-insns32.py write them.
enum {
    RCX_SMALL = 1 << 0, R8_SMALL = 1 << 1, BITOFF = 1 << 2, POPF = 1 << 3, XLAT = 1 << 4,
    APPROX = 1 << 5, NOMEM = 1 << 6, SSE = 1 << 7, RCX17 = 1 << 8, DIV = 1 << 9,
    // 32-bit snippets
    X87 = 1 << 10,      // random x87 stack, a few registers in use
    MMX = 1 << 11,      // random MMX registers
    X87RC = 1 << 12,    // random x87 rounding control
    FSWAX = 1 << 13,    // AX (and the like) holds the status word: compare C0 C2 C3 TOP only
    CMPC1 = 1 << 14,    // compare C1 too
    X87INT = 1 << 15,
    X87NEAR = 1 << 22,  // x87 values within 2^20 of 1 (FPREM reduces them in one step)
    X87UNIT = 1 << 23,  // x87 values within 0.25 of 0 (F2XM1 and FYL2XP1's domain)
};

// Every rnd() call is its own statement: the order C evaluates a function's
// arguments in differs between compilers, and both builds must agree.

struct Template { const uint8_t *start, *end; const char *name; uint64_t undef, flags; };

struct NState {
    uint64_t gpr[16];
    uint64_t flags;
    uint32_t mxcsr, pad;
    uint8_t gap[0x100 - 0x90];
    X64Vec xmm[16];
};

#define IN  ((struct NState *)ANCHOR)
#define OUT ((struct NState *)(ANCHOR + 0x400))
#define COUNT(a) (sizeof(a) / sizeof((a)[0]))

static uint64_t rng;
static uint64_t rnd(void)
{
    rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
    return rng * 0x2545F4914F6CDD1Dull;
}

static const uint64_t special64[] = {
    0, 1, 2, 3, 0x7f, 0x80, 0xff, 0x100, 0x7fff, 0x8000, 0xffff, 0x10000, 0x7fffffff, 0x80000000,
    0xffffffff, 0x100000000, 0x7fffffffffffffff, 0x8000000000000000, 0xffffffffffffffff, 0xfffffffffffffffe,
    0x8000000000000001, 0x00000000ffffff80, 0xffffffff80000000, 0x5555555555555555, 0xaaaaaaaaaaaaaaaa,
};
static const uint32_t special32f[] = {
    0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x3f000000, 0x3fc00000, 0x40200000, 0xc0200000,
    0x7f800000, 0xff800000, 0x7fc00000, 0x7fa00000, 0xffc00001, 0x7fc12345, 0x00000001, 0x007fffff,
    0x80000010, 0x7f7fffff, 0x00800000, 0x4f000000, 0xcf000000, 0x4effffff, 0x501502f9, 0x40400000,
    0x3f800001, 0xdf000000, 0x5f000000, 0x3effffff, 0x00400000, 0x0f800000,
};
static const uint64_t special64f[] = {
    0x0000000000000000, 0x8000000000000000, 0x3ff0000000000000, 0xbff0000000000000, 0x3fe0000000000000,
    0x3ff8000000000000, 0x4004000000000000, 0xc004000000000000, 0x7ff0000000000000, 0xfff0000000000000,
    0x7ff8000000000000, 0x7ff4000000000000, 0xfff8000000000123, 0x0000000000000001, 0x000fffffffffffff,
    0x0010000000000000, 0x7fefffffffffffff, 0x43e0000000000000, 0xc3e0000000000000, 0x41dfffffffc00000,
    0x41e0000000000000, 0xc1e0000000000000, 0x41dfffffffe00000, 0x3810000000000000, 0x47efffffe0000000,
    0x36a0000000000000, 0x3fefffffffffffff, 0x4330000000000001,
};

static uint64_t rnd_int(void)
{
    switch (rnd() % 8) {
    case 0: case 1: return special64[rnd() % COUNT(special64)];
    case 2: return rnd() & 0xff;
    case 3: return rnd() & 0xffff;
    case 4: return rnd() & 0xffffffff;
    case 5: return (uint64_t)(int64_t)(int32_t)rnd();
    case 6: return (uint64_t)((int64_t)(rnd() % 9) - 4);
    default: return rnd();
    }
}

static void rnd_vec(X64Vec *v)
{
    switch (rnd() % 3) {
    case 0:
        for (int i = 0; i < 4; i++)
            v->d[i] = rnd() % 3 ? special32f[rnd() % COUNT(special32f)] : (uint32_t)rnd();
        break;
    case 1:
        for (int i = 0; i < 2; i++)
            v->q[i] = rnd() % 3 ? special64f[rnd() % COUNT(special64f)] : rnd();
        break;
    default:
        v->q[0] = rnd_int(); v->q[1] = rnd_int();
        break;
    }
}

static bool approx_equal(uint32_t a, uint32_t b)
{
    float fa, fb;
    memcpy(&fa, &a, 4); memcpy(&fb, &b, 4);
    if (a == b) return true;
    if (isnan(fa) && isnan(fb)) return true;
    if (fa == 0 || fb == 0) return (fa == 0) == (fb == 0) && fabsf(fa - fb) < 1e-30f;
    return fabsf(fa - fb) <= fabsf(fb) * (1.0f / 2048);
}

static uint64_t hash(const void *p, size_t n)
{
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) h = (h ^ ((const uint8_t *)p)[i]) * 0x100000001b3ull;
    return h;
}

static sigjmp_buf fault_jump;
static volatile int fault_signal;
static void on_fault(int sig) { fault_signal = sig; siglongjmp(fault_jump, 1); }

static bool map_fixed(unsigned long address, size_t size, int prot)
{
    return mmap((void *)address, size, prot, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0) == (void *)address;
}

static void catch_faults(void)
{
    static uint8_t altstack[65536];
    stack_t ss = { .ss_sp = altstack, .ss_size = sizeof altstack };
    sigaltstack(&ss, NULL);
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_fault;
    sa.sa_flags = SA_ONSTACK | SA_NODEFER;
    int sigs[] = { SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGTRAP };
    for (size_t i = 0; i < COUNT(sigs); i++) sigaction(sigs[i], &sa, NULL);
}


// --- 32-bit snippets ---------------------------------------------------------------

// The state block the 32-bit prologue loads and the epilogue stores.
struct NState32 {
    uint32_t gpr[8];        // eax ecx edx ebx esp ebp esi edi
    uint32_t eflags;
    uint32_t pad[7];
    uint8_t fx[512];        // FXSAVE image: x87, MMX, XMM0-7, MXCSR (at 0x40: 16-byte aligned)
};
#define IN32  ((struct NState32 *)ANCHOR)
#define OUT32 ((struct NState32 *)(ANCHOR + 0x400))

static void put_f80(uint8_t *slot, uint64_t mantissa, uint16_t exponent)
{
    memset(slot, 0, 16);
    memcpy(slot, &mantissa, 8);
    memcpy(slot + 8, &exponent, 2);
}

// A double in a range where x87 arithmetic at 53-bit precision and double
// arithmetic agree, as an 80-bit register.
static void rnd_x87(uint8_t *slot, uint64_t flags)
{
    static const uint64_t special[] = {
        0x0000000000000000, 0x8000000000000000, 0x3ff0000000000000, 0xbff0000000000000, 0x3fe0000000000000,
        0x4004000000000000, 0xc004000000000000, 0x4008000000000000, 0x4024000000000000, 0x7ff0000000000000,
        0xfff0000000000000, 0x7ff8000000000000, 0x3fb999999999999a, 0x3fd5555555555555, 0x41e0000000000000,
        0xc1e0000000000000, 0x3ff8000000000000, 0xbfe8000000000000, 0x4012000000000000, 0x40c81cd6c8b43958,
        0x4059000000000000, 0xc08f400000000000, 0x3ff0000000000001, 0x433fffffffffffff,
    };
    uint64_t bits, low, exponent;
    unsigned kind = (unsigned)(rnd() % 6);
    if (flags & (X87NEAR | X87UNIT)) kind = 3;
    switch (kind) {
    case 0: case 1: bits = special[rnd() % COUNT(special)]; break;
    case 2: { double v = (double)((int64_t)(rnd() % 201) - 100); memcpy(&bits, &v, 8); break; }
    default:
        low = rnd() & 0x800fffffffffffffull;
        exponent = rnd();
        exponent = (flags & X87UNIT) ? 1023 - 30 + exponent % 28 : (flags & X87NEAR) ? 1023 - 20 + exponent % 41 : 1023 - 40 + exponent % 81;
        bits = low | (exponent << 52);
        break;
    }
    uint16_t sign = (uint16_t)((bits >> 48) & 0x8000);
    int e = (int)((bits >> 52) & 0x7ff);
    uint64_t frac = bits & 0x000fffffffffffffull;
    if (e == 0x7ff) put_f80(slot, 0x8000000000000000ull | (frac << 11), sign | 0x7fff);
    else if (e) put_f80(slot, 0x8000000000000000ull | (frac << 11), (uint16_t)(sign | (e - 1023 + 16383)));
    else put_f80(slot, 0, sign);
}

static void setup_input32(uint64_t flags)
{
    struct NState32 *in = IN32;
    for (int i = 0; i < 8; i++) in->gpr[i] = (uint32_t)rnd_int();
    in->eflags = 0x202 | (uint32_t)(rnd() & 0xcd5);
    in->gpr[6] = (uint32_t)(MEM + 0x1000);   // esi
    in->gpr[7] = (uint32_t)(MEM + 0x2000);   // edi
    in->gpr[4] = (uint32_t)(STACK + 0x1000); // esp
    if (flags & RCX_SMALL) in->gpr[1] = (uint32_t)(rnd() % 41);
    if (flags & RCX17) in->gpr[1] = (uint32_t)(rnd() % 17);
    if (flags & BITOFF) in->gpr[3] = (uint32_t)((int32_t)(rnd() % 401) - 200);
    if (flags & XLAT) in->gpr[3] = (uint32_t)(MEM + 0x1800);
    if (flags & POPF) in->gpr[3] = (uint32_t)((rnd() & 0x200ed5) | 2);

    uint8_t *fx = in->fx;
    memset(fx, 0, 512);
    uint16_t fcw = 0x027f, fsw;
    if (flags & X87RC) fcw = (uint16_t)(fcw | ((rnd() % 4) << 10));
    int top = (int)(rnd() % 8);
    int used = (flags & MMX) ? 8 : 2 + (int)(rnd() % 4);
    fsw = (uint16_t)((top << 11) | (rnd() & 0x4700));
    uint8_t tags = 0;
    for (int i = 0; i < used; i++) tags |= (uint8_t)(1 << ((top + i) & 7));
    memcpy(fx, &fcw, 2);
    memcpy(fx + 2, &fsw, 2);
    fx[4] = tags;
    for (int i = 0; i < 8; i++) {
        if (flags & MMX) {
            uint64_t mantissa = rnd();
            uint16_t exponent = (uint16_t)rnd();
            put_f80(fx + 32 + 16 * i, mantissa, exponent);
        } else {
            rnd_x87(fx + 32 + 16 * i, flags);
        }
    }
    uint32_t mxcsr = 0x1f80;
    if (flags & SSE) {
        mxcsr |= (uint32_t)(rnd() % 4) << 13;
        if (rnd() % 4 == 0) mxcsr |= 0x8000;
        if (rnd() % 4 == 0) mxcsr |= 0x40;
    }
    memcpy(fx + 24, &mxcsr, 4);
    for (int i = 0; i < 8; i++) {
        X64Vec v;
        rnd_vec(&v);
        memcpy(fx + 160 + 16 * i, &v, 16);
    }
    for (size_t i = 0; i < MEMSZ; i += 8) { uint64_t v = rnd_int(); memcpy((uint8_t *)MEM + i, &v, 8); }
    for (size_t i = 0; i < STACKSZ; i += 8) { uint64_t v = rnd(); memcpy((uint8_t *)STACK + i, &v, 8); }
    if (flags & DIV) {
        int size = (int)((flags >> 16) & 15);
        bool is_signed = (flags >> 20) & 1, in_mem = (flags >> 21) & 1;
        uint32_t m = size == 4 ? 0xffffffffu : (1u << (size * 8)) - 1;
        uint32_t divisor = (uint32_t)rnd_int() & m;
        if (!divisor) divisor = 7;
        if (is_signed) {
            uint32_t low = in->gpr[0] & m;
            int32_t sl = size == 4 ? (int32_t)low : size == 2 ? (int16_t)low : (int8_t)low;
            int32_t sd = size == 4 ? (int32_t)divisor : size == 2 ? (int16_t)divisor : (int8_t)divisor;
            if (sd == -1 && (uint32_t)sl == ((1u << (size * 8 - 1)) | ~m)) divisor = 3;
            uint32_t high = sl < 0 ? m : 0;
            if (size == 1) in->gpr[0] = (in->gpr[0] & ~0xffffu) | (high << 8) | low;
            else in->gpr[2] = (in->gpr[2] & ~m) | high;
        } else {
            uint32_t high = (uint32_t)rnd_int() % divisor;
            if (size == 1) in->gpr[0] = (in->gpr[0] & ~0xff00u) | (high << 8);
            else in->gpr[2] = (in->gpr[2] & ~m) | high;
        }
        if (in_mem) memcpy((uint8_t *)MEM + 0x1000, &divisor, (size_t)size);
        else in->gpr[3] = (in->gpr[3] & ~m) | divisor;
    }
}

#endif
