// SPDX-License-Identifier: GPL-3.0-or-later
// Madeira Converter Exception: see LICENSE-EXCEPTION.md
//
// An x86-64 interpreter that runs guest code without executable memory. See
// X64Interpreter.h for the interface and docs/NO-JIT.md for where it fits.
//
// Each instruction is decoded from guest memory and carried out here; nothing
// is cached, so self-modifying code and code a guest JIT writes need no
// special handling. What is implemented: the general-purpose instruction set
// (with LOCK as real host atomics), string instructions, SSE, SSE2, SSE3 and
// PSHUFB/PALIGNR, CMPXCHG16B, POPCNT/LZCNT/TZCNT, CPUID with a fixed profile,
// RDTSC, and the x87 control-word instructions C runtimes run at start-up.
// What is not: x87 arithmetic, MMX, AVX and anything VEX or EVEX encoded,
// SSSE3 beyond PSHUFB/PALIGNR, SSE4, and system instructions. Those stop the
// run with X64_EXIT_UNSUPPORTED (or X64_EXIT_PRIVILEGED) and leave the state
// as it was before the instruction, so a caller can report or handle them.
//
// Floating point follows x86 rather than the host: NaN propagation and the
// default NaN, MXCSR rounding (through the host rounding mode), DAZ and FTZ,
// and the "integer indefinite" result of out-of-range conversions. MXCSR's
// exception flags are not updated and unmasked exceptions are not raised.
//
// Memory accesses are host loads and stores. With X64_INTERP_TSO (the
// default) aligned loads are acquire and aligned stores release, which keeps
// x86's store ordering on a weakly ordered host such as ARM64.

#if !defined(__APPLE__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L   // clock_gettime under -std=c11
#endif

#include "X64Interpreter.h"

#include <fenv.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__clang__)
#pragma STDC FENV_ACCESS ON
#endif

#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "X64Interpreter needs a little-endian host"
#endif

#ifndef X64_INTERP_TSO
#define X64_INTERP_TSO 1
#endif

#define FL_CF 0x0001ull
#define FL_PF 0x0004ull
#define FL_AF 0x0010ull
#define FL_ZF 0x0040ull
#define FL_SF 0x0080ull
#define FL_IF 0x0200ull
#define FL_DF 0x0400ull
#define FL_OF 0x0800ull
#define FL_AC 0x40000ull
#define FL_ID 0x200000ull
#define FL_ARITH (FL_CF | FL_PF | FL_AF | FL_ZF | FL_SF | FL_OF)
// What POPF may change in user mode (IF and IOPL stay; TF is not emulated).
#define FL_POPF (FL_ARITH | FL_DF | FL_AC | FL_ID)

#define MXCSR_DAZ 0x0040u
#define MXCSR_FTZ 0x8000u

#define CONT (-1)

// ---------------------------------------------------------------------------
// Decoding

typedef struct Dec {
    uint64_t start, next;
    const uint8_t *begin, *p;
    uint8_t rex;
    uint8_t rep;            // 0, 0xF2 or 0xF3: the last of them
    bool opsize, addrsize, lock;
    int seg;                // -1, 4 (FS) or 5 (GS)
    // ModRM
    uint8_t mod, regf, rmf;
    int reg, rm;            // with REX.R and REX.B
    bool mem;
    int base, index, scale;
    int64_t disp;
    bool riprel;
    uint64_t off, ea;       // offset without and with the segment base
} Dec;

static inline uint8_t fetch8(Dec *d) { return *d->p++; }
static inline uint16_t fetch16(Dec *d) { uint16_t v; memcpy(&v, d->p, 2); d->p += 2; return v; }
static inline uint32_t fetch32(Dec *d) { uint32_t v; memcpy(&v, d->p, 4); d->p += 4; return v; }
static inline uint64_t fetch64(Dec *d) { uint64_t v; memcpy(&v, d->p, 8); d->p += 8; return v; }

static void modrm(Dec *d)
{
    uint8_t m = fetch8(d);
    d->mod = m >> 6;
    d->regf = (m >> 3) & 7;
    d->rmf = m & 7;
    d->reg = d->regf | ((d->rex & 4) << 1);
    if (d->mod == 3) {
        d->mem = false;
        d->rm = d->rmf | ((d->rex & 1) << 3);
        return;
    }
    d->mem = true;
    d->base = -1;
    d->index = -1;
    d->scale = 0;
    d->disp = 0;
    d->riprel = false;
    if (d->rmf == 4) {
        uint8_t sib = fetch8(d);
        int index = ((sib >> 3) & 7) | ((d->rex & 2) << 2);
        if (index != 4) { d->index = index; d->scale = sib >> 6; }
        if ((sib & 7) == 5 && d->mod == 0)
            d->disp = (int32_t)fetch32(d);
        else
            d->base = (sib & 7) | ((d->rex & 1) << 3);
    } else if (d->rmf == 5 && d->mod == 0) {
        d->riprel = true;
        d->disp = (int32_t)fetch32(d);
    } else {
        d->base = d->rmf | ((d->rex & 1) << 3);
    }
    if (d->mod == 1) d->disp = (int8_t)fetch8(d);
    else if (d->mod == 2) d->disp = (int32_t)fetch32(d);
}

// Called once every byte of the instruction has been fetched.
static bool finish(const X64State *s, Dec *d)
{
    size_t length = (size_t)(d->p - d->begin);
    if (length > 15) return false;
    d->next = d->start + length;
    if (d->mem) {
        uint64_t off = (uint64_t)d->disp;
        if (d->riprel) off += d->next;
        if (d->base >= 0) off += s->gpr[d->base];
        if (d->index >= 0) off += s->gpr[d->index] << d->scale;
        if (d->addrsize) off &= 0xffffffffull;
        d->off = off;
        d->ea = off + (d->seg == 4 ? s->fs_base : d->seg == 5 ? s->gs_base : 0);
    }
    return true;
}

static inline int osz(const Dec *d) { return (d->rex & 8) ? 8 : d->opsize ? 2 : 4; }

// ---------------------------------------------------------------------------
// Registers and memory

static inline uint64_t mask_of(int size) { return size == 8 ? ~0ull : (1ull << (size * 8)) - 1; }
static inline uint64_t sign_of(int size) { return 1ull << (size * 8 - 1); }

static inline int64_t sext(uint64_t v, int size)
{
    switch (size) {
    case 1: return (int8_t)v;
    case 2: return (int16_t)v;
    case 4: return (int32_t)v;
    default: return (int64_t)v;
    }
}

static inline uint64_t get_reg(const X64State *s, const Dec *d, int reg, int size)
{
    if (size == 1) {
        if (!d->rex && reg >= 4 && reg < 8) return (s->gpr[reg - 4] >> 8) & 0xff;
        return s->gpr[reg] & 0xff;
    }
    return s->gpr[reg] & mask_of(size);
}

static inline void set_reg(X64State *s, const Dec *d, int reg, int size, uint64_t v)
{
    switch (size) {
    case 1:
        if (!d->rex && reg >= 4 && reg < 8)
            s->gpr[reg - 4] = (s->gpr[reg - 4] & ~0xff00ull) | ((v & 0xff) << 8);
        else
            s->gpr[reg] = (s->gpr[reg] & ~0xffull) | (v & 0xff);
        break;
    case 2: s->gpr[reg] = (s->gpr[reg] & ~0xffffull) | (v & 0xffff); break;
    case 4: s->gpr[reg] = v & 0xffffffffull; break;
    default: s->gpr[reg] = v; break;
    }
}

static inline uint64_t mem_read(uint64_t addr, int size)
{
    const void *p = (const void *)(uintptr_t)addr;
#if X64_INTERP_TSO
    switch (size) {
    case 1: return __atomic_load_n((const uint8_t *)p, __ATOMIC_ACQUIRE);
    case 2: if (!(addr & 1)) return __atomic_load_n((const uint16_t *)p, __ATOMIC_ACQUIRE); break;
    case 4: if (!(addr & 3)) return __atomic_load_n((const uint32_t *)p, __ATOMIC_ACQUIRE); break;
    case 8: if (!(addr & 7)) return __atomic_load_n((const uint64_t *)p, __ATOMIC_ACQUIRE); break;
    }
#endif
    uint64_t v = 0;
    memcpy(&v, p, (size_t)size);
#if X64_INTERP_TSO
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
#endif
    return v;
}

static inline void mem_write(uint64_t addr, int size, uint64_t v)
{
    void *p = (void *)(uintptr_t)addr;
#if X64_INTERP_TSO
    switch (size) {
    case 1: __atomic_store_n((uint8_t *)p, (uint8_t)v, __ATOMIC_RELEASE); return;
    case 2: if (!(addr & 1)) { __atomic_store_n((uint16_t *)p, (uint16_t)v, __ATOMIC_RELEASE); return; } break;
    case 4: if (!(addr & 3)) { __atomic_store_n((uint32_t *)p, (uint32_t)v, __ATOMIC_RELEASE); return; } break;
    case 8: if (!(addr & 7)) { __atomic_store_n((uint64_t *)p, v, __ATOMIC_RELEASE); return; } break;
    }
    __atomic_thread_fence(__ATOMIC_RELEASE);
#endif
    memcpy(p, &v, (size_t)size);
}

static inline void vec_read(X64Vec *v, uint64_t addr, int bytes)
{
    memset(v, 0, sizeof *v);
    memcpy(v, (const void *)(uintptr_t)addr, (size_t)bytes);
#if X64_INTERP_TSO
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
#endif
}

static inline void vec_write(uint64_t addr, const X64Vec *v, int bytes)
{
#if X64_INTERP_TSO
    __atomic_thread_fence(__ATOMIC_RELEASE);
#endif
    memcpy((void *)(uintptr_t)addr, v, (size_t)bytes);
}

// LOCKed operations. An unaligned one (an x86 "split lock", rare and slow there
// too) faults on an ARM64 host where it crosses a 16-byte boundary.
static bool mem_cas(uint64_t addr, int size, uint64_t *expected, uint64_t desired)
{
    void *p = (void *)(uintptr_t)addr;
    bool ok;
    switch (size) {
    case 1: { uint8_t e = (uint8_t)*expected;
        ok = __atomic_compare_exchange_n((uint8_t *)p, &e, (uint8_t)desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
        *expected = e; return ok; }
    case 2: { uint16_t e = (uint16_t)*expected;
        ok = __atomic_compare_exchange_n((uint16_t *)p, &e, (uint16_t)desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
        *expected = e; return ok; }
    case 4: { uint32_t e = (uint32_t)*expected;
        ok = __atomic_compare_exchange_n((uint32_t *)p, &e, (uint32_t)desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
        *expected = e; return ok; }
    default:
        return __atomic_compare_exchange_n((uint64_t *)p, expected, desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    }
}

static uint64_t mem_xchg(uint64_t addr, int size, uint64_t v)
{
    void *p = (void *)(uintptr_t)addr;
    switch (size) {
    case 1: return __atomic_exchange_n((uint8_t *)p, (uint8_t)v, __ATOMIC_SEQ_CST);
    case 2: return __atomic_exchange_n((uint16_t *)p, (uint16_t)v, __ATOMIC_SEQ_CST);
    case 4: return __atomic_exchange_n((uint32_t *)p, (uint32_t)v, __ATOMIC_SEQ_CST);
    default: return __atomic_exchange_n((uint64_t *)p, v, __ATOMIC_SEQ_CST);
    }
}

static inline uint64_t get_rm(const X64State *s, const Dec *d, int size)
{
    return d->mem ? mem_read(d->ea, size) : get_reg(s, d, d->rm, size);
}

static inline void set_rm(X64State *s, const Dec *d, int size, uint64_t v)
{
    if (d->mem) mem_write(d->ea, size, v);
    else set_reg(s, d, d->rm, size, v);
}

static inline void push(X64State *s, uint64_t v, int size)
{
    s->gpr[X64_RSP] -= (uint64_t)size;
    mem_write(s->gpr[X64_RSP], size, v);
}

static inline uint64_t pop(X64State *s, int size)
{
    uint64_t v = mem_read(s->gpr[X64_RSP], size);
    s->gpr[X64_RSP] += (uint64_t)size;
    return v;
}

static inline uint64_t seg_base(const X64State *s, const Dec *d)
{
    return d->seg == 4 ? s->fs_base : d->seg == 5 ? s->gs_base : 0;
}

// ---------------------------------------------------------------------------
// Flags and integer arithmetic

static inline uint64_t parity(uint64_t r)
{
    return (__builtin_popcount((unsigned)(r & 0xff)) & 1) ? 0 : FL_PF;
}

static inline uint64_t szp(uint64_t r, int size)
{
    r &= mask_of(size);
    return (r == 0 ? FL_ZF : 0) | ((r & sign_of(size)) ? FL_SF : 0) | parity(r);
}

static inline void set_flags(X64State *s, uint64_t which, uint64_t value)
{
    s->rflags = (s->rflags & ~which) | (value & which);
}

static uint64_t op_add(X64State *s, uint64_t a, uint64_t b, uint64_t carry, int size)
{
    uint64_t m = mask_of(size);
    a &= m; b &= m;
    uint64_t r = (a + b + carry) & m;
    bool cf = size == 8 ? (r < a || (carry && r == a)) : (((a + b + carry) >> (size * 8)) & 1);
    uint64_t f = (cf ? FL_CF : 0) | (((a ^ r) & (b ^ r) & sign_of(size)) ? FL_OF : 0)
               | (((a ^ b ^ r) & 0x10) ? FL_AF : 0) | szp(r, size);
    set_flags(s, FL_ARITH, f);
    return r;
}

static uint64_t op_sub(X64State *s, uint64_t a, uint64_t b, uint64_t borrow, int size)
{
    uint64_t m = mask_of(size);
    a &= m; b &= m;
    uint64_t r = (a - b - borrow) & m;
    bool cf = a < b || (borrow && a == b);
    uint64_t f = (cf ? FL_CF : 0) | (((a ^ b) & (a ^ r) & sign_of(size)) ? FL_OF : 0)
               | (((a ^ b ^ r) & 0x10) ? FL_AF : 0) | szp(r, size);
    set_flags(s, FL_ARITH, f);
    return r;
}

enum { ALU_ADD, ALU_OR, ALU_ADC, ALU_SBB, ALU_AND, ALU_SUB, ALU_XOR, ALU_CMP };

// `carry` is CF as it was before the instruction (ADC and SBB).
static uint64_t alu(X64State *s, int op, uint64_t a, uint64_t b, uint64_t carry, int size)
{
    uint64_t r;
    switch (op) {
    case ALU_ADD: return op_add(s, a, b, 0, size);
    case ALU_ADC: return op_add(s, a, b, carry, size);
    case ALU_SBB: return op_sub(s, a, b, carry, size);
    case ALU_SUB: case ALU_CMP: return op_sub(s, a, b, 0, size);
    case ALU_OR: r = a | b; break;
    case ALU_AND: r = a & b; break;
    default: r = a ^ b; break;
    }
    r &= mask_of(size);
    set_flags(s, FL_ARITH, szp(r, size));
    return r;
}

static bool cond(const X64State *s, int cc)
{
    uint64_t f = s->rflags;
    bool r;
    switch (cc >> 1) {
    case 0: r = f & FL_OF; break;
    case 1: r = f & FL_CF; break;
    case 2: r = f & FL_ZF; break;
    case 3: r = (f & FL_CF) || (f & FL_ZF); break;
    case 4: r = f & FL_SF; break;
    case 5: r = f & FL_PF; break;
    case 6: r = !(f & FL_SF) != !(f & FL_OF); break;
    default: r = (f & FL_ZF) || (!(f & FL_SF) != !(f & FL_OF)); break;
    }
    return (cc & 1) ? !r : r;
}

// Read-modify-write operations, which LOCK makes atomic.
enum { RMW_ALU, RMW_INC, RMW_DEC, RMW_NOT, RMW_NEG, RMW_BTS, RMW_BTR, RMW_BTC };

static uint64_t rmw_compute(X64State *s, int kind, int op, uint64_t a, uint64_t b, uint64_t carry, int size)
{
    uint64_t m = mask_of(size);
    switch (kind) {
    case RMW_ALU: return alu(s, op, a, b, carry, size);
    case RMW_INC: { uint64_t cf = s->rflags & FL_CF; uint64_t r = op_add(s, a, 1, 0, size);
        set_flags(s, FL_CF, cf); return r; }
    case RMW_DEC: { uint64_t cf = s->rflags & FL_CF; uint64_t r = op_sub(s, a, 1, 0, size);
        set_flags(s, FL_CF, cf); return r; }
    case RMW_NOT: return ~a & m;
    case RMW_NEG: { uint64_t r = op_sub(s, 0, a, 0, size);
        set_flags(s, FL_CF, (a & m) ? FL_CF : 0); return r; }
    default: {
        uint64_t bit = 1ull << b;
        set_flags(s, FL_CF, (a & bit) ? FL_CF : 0);
        if (kind == RMW_BTS) return a | bit;
        if (kind == RMW_BTR) return a & ~bit;
        return a ^ bit;
    }
    }
}

static void rmw(X64State *s, const Dec *d, int size, int kind, int op, uint64_t b)
{
    uint64_t carry = s->rflags & FL_CF;
    if (d->mem && d->lock) {
        uint64_t old = mem_read(d->ea, size);
        for (;;) {
            uint64_t v = rmw_compute(s, kind, op, old, b, carry, size);
            if (mem_cas(d->ea, size, &old, v)) return;
        }
    }
    set_rm(s, d, size, rmw_compute(s, kind, op, get_rm(s, d, size), b, carry, size));
}

static uint64_t shift(X64State *s, int op, uint64_t a, unsigned count, int size)
{
    int bits = size * 8;
    uint64_t m = mask_of(size), sign = sign_of(size);
    unsigned cnt = count & (size == 8 ? 63 : 31);
    a &= m;
    if (cnt == 0) return a;
    uint64_t r;
    bool cf, of;
    switch (op) {
    case 0: { // ROL
        unsigned c = cnt % (unsigned)bits;
        r = c ? ((a << c) | (a >> (bits - (int)c))) & m : a;
        cf = r & 1;
        of = ((r & sign) != 0) != cf;
        set_flags(s, FL_CF | FL_OF, (cf ? FL_CF : 0) | (of ? FL_OF : 0));
        return r;
    }
    case 1: { // ROR
        unsigned c = cnt % (unsigned)bits;
        r = c ? ((a >> c) | (a << (bits - (int)c))) & m : a;
        cf = (r & sign) != 0;
        of = cf != ((r & (sign >> 1)) != 0);
        set_flags(s, FL_CF | FL_OF, (cf ? FL_CF : 0) | (of ? FL_OF : 0));
        return r;
    }
    case 2: case 3: { // RCL, RCR
        unsigned c = size < 4 ? cnt % (unsigned)(bits + 1) : cnt;
        uint64_t carry = s->rflags & FL_CF;
        r = a;
        for (unsigned i = 0; i < c; i++) {
            uint64_t out;
            if (op == 2) { out = (r & sign) != 0; r = ((r << 1) | carry) & m; }
            else { out = r & 1; r = (r >> 1) | (carry ? sign : 0); }
            carry = out;
        }
        if (op == 2) of = ((r & sign) != 0) != (carry != 0);
        else of = ((r & sign) != 0) != ((r & (sign >> 1)) != 0);
        set_flags(s, FL_CF | FL_OF, (carry ? FL_CF : 0) | (of ? FL_OF : 0));
        return r;
    }
    case 4: case 6: // SHL
        r = (a << cnt) & m;
        cf = cnt <= (unsigned)bits ? (a >> (bits - (int)cnt)) & 1 : 0;
        of = ((r & sign) != 0) != cf;
        break;
    case 5: // SHR
        r = cnt < (unsigned)bits ? a >> cnt : 0;
        cf = cnt <= (unsigned)bits ? (a >> (cnt - 1)) & 1 : 0;
        of = (a & sign) != 0;
        break;
    default: { // SAR
        int64_t sa = sext(a, size);
        r = (uint64_t)(sa >> (cnt < (unsigned)bits ? cnt : (unsigned)bits - 1)) & m;
        cf = (sa >> (cnt - 1 < (unsigned)bits ? cnt - 1 : (unsigned)bits - 1)) & 1;
        of = false;
        break;
    }
    }
    set_flags(s, FL_ARITH, (cf ? FL_CF : 0) | (of ? FL_OF : 0) | szp(r, size));
    return r;
}

// SHLD (left) and SHRD: `a` is the destination, `b` supplies the bits shifted in.
static uint64_t double_shift(X64State *s, bool left, uint64_t a, uint64_t b, unsigned count, int size)
{
    int bits = size * 8;
    uint64_t m = mask_of(size), sign = sign_of(size);
    unsigned cnt = count & (size == 8 ? 63 : 31);
    a &= m; b &= m;
    if (cnt == 0) return a;
    uint64_t r;
    bool cf;
    if (size == 2 && cnt > 16) {
        // Undefined on x86; a 48-bit funnel keeps the result deterministic.
        uint64_t x = (a << 32) | (b << 16) | a;
        r = left ? (x << cnt) >> 32 & m : (x >> cnt) & m;
        cf = left ? (x >> (48 - cnt)) & 1 : (x >> (cnt - 1)) & 1;
    } else if (left) {
        r = ((a << cnt) | b >> (bits - (int)cnt)) & m;
        cf = (a >> (bits - (int)cnt)) & 1;
    } else {
        r = ((a >> cnt) | b << (bits - (int)cnt)) & m;
        cf = (a >> (cnt - 1)) & 1;
    }
    bool of = ((r ^ a) & sign) != 0;
    set_flags(s, FL_ARITH, (cf ? FL_CF : 0) | (of ? FL_OF : 0) | szp(r, size));
    return r;
}

// ---------------------------------------------------------------------------
// Floating point (SSE)

static inline float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline double u2d(uint64_t u) { double f; memcpy(&f, &u, 8); return f; }
static inline uint64_t d2u(double f) { uint64_t u; memcpy(&u, &f, 8); return u; }

#define F32_INDEFINITE 0xffc00000u
#define F64_INDEFINITE 0xfff8000000000000ull

static inline bool f32_nan(uint32_t x) { return (x & 0x7f800000u) == 0x7f800000u && (x & 0x007fffffu); }
static inline bool f64_nan(uint64_t x) { return (x & 0x7ff0000000000000ull) == 0x7ff0000000000000ull && (x & 0x000fffffffffffffull); }
static inline bool f32_denormal(uint32_t x) { return !(x & 0x7f800000u) && (x & 0x007fffffu); }
static inline bool f64_denormal(uint64_t x) { return !(x & 0x7ff0000000000000ull) && (x & 0x000fffffffffffffull); }

static inline uint32_t f32_in(const X64State *s, uint32_t x)
{
    return ((s->mxcsr & MXCSR_DAZ) && f32_denormal(x)) ? x & 0x80000000u : x;
}
static inline uint64_t f64_in(const X64State *s, uint64_t x)
{
    return ((s->mxcsr & MXCSR_DAZ) && f64_denormal(x)) ? x & 0x8000000000000000ull : x;
}
static inline uint32_t f32_out(const X64State *s, uint32_t x)
{
    if (f32_nan(x)) return F32_INDEFINITE;  // an invalid operation; NaN inputs never get here
    return ((s->mxcsr & MXCSR_FTZ) && f32_denormal(x)) ? x & 0x80000000u : x;
}
static inline uint64_t f64_out(const X64State *s, uint64_t x)
{
    if (f64_nan(x)) return F64_INDEFINITE;
    return ((s->mxcsr & MXCSR_FTZ) && f64_denormal(x)) ? x & 0x8000000000000000ull : x;
}

// The arithmetic ones are the opcode less 0x58 (58 ADD, 59 MUL, 5C SUB, 5D MIN, 5E DIV, 5F MAX).
enum { FOP_ADD = 0, FOP_MUL = 1, FOP_SUB = 4, FOP_MIN = 5, FOP_DIV = 6, FOP_MAX = 7, FOP_SQRT, FOP_RSQRT, FOP_RCP };

// `a` is the destination (first source) and `b` the source operand.
static uint32_t f32_op(const X64State *s, int op, uint32_t a, uint32_t b)
{
    a = f32_in(s, a);
    b = f32_in(s, b);
    float fa = u2f(a), fb = u2f(b);
    switch (op) {
    case FOP_MIN: return fa < fb ? a : b;   // NaN or equal: the source, as is
    case FOP_MAX: return fa > fb ? a : b;
    case FOP_SQRT:
        if (f32_nan(b)) return b | 0x00400000u;
        return f32_out(s, f2u(sqrtf(fb)));
    case FOP_RCP: case FOP_RSQRT:
        // Hardware gives a 12-bit approximation; this is the exact answer rounded,
        // which is within its error bound. Denormal inputs and tiny results are zero.
        if (f32_nan(b)) return b | 0x00400000u;
        if (f32_denormal(b)) b &= 0x80000000u;
        if (op == FOP_RCP) {
            uint32_t r = f2u(1.0f / u2f(b));
            return f32_denormal(r) ? r & 0x80000000u : r;
        }
        if ((b & 0x80000000u) && (b & 0x7fffffffu)) return F32_INDEFINITE;
        return f2u(1.0f / sqrtf(u2f(b)));
    }
    if (f32_nan(a)) return a | 0x00400000u;
    if (f32_nan(b)) return b | 0x00400000u;
    volatile float r;
    switch (op) {
    case FOP_ADD: r = fa + fb; break;
    case FOP_SUB: r = fa - fb; break;
    case FOP_MUL: r = fa * fb; break;
    default: r = fa / fb; break;
    }
    return f32_out(s, f2u(r));
}

static uint64_t f64_op(const X64State *s, int op, uint64_t a, uint64_t b)
{
    a = f64_in(s, a);
    b = f64_in(s, b);
    double fa = u2d(a), fb = u2d(b);
    switch (op) {
    case FOP_MIN: return fa < fb ? a : b;
    case FOP_MAX: return fa > fb ? a : b;
    case FOP_SQRT:
        if (f64_nan(b)) return b | 0x0008000000000000ull;
        return f64_out(s, d2u(sqrt(fb)));
    }
    if (f64_nan(a)) return a | 0x0008000000000000ull;
    if (f64_nan(b)) return b | 0x0008000000000000ull;
    volatile double r;
    switch (op) {
    case FOP_ADD: r = fa + fb; break;
    case FOP_SUB: r = fa - fb; break;
    case FOP_MUL: r = fa * fb; break;
    default: r = fa / fb; break;
    }
    return f64_out(s, d2u(r));
}

static uint64_t f32_to_f64(const X64State *s, uint32_t x)
{
    x = f32_in(s, x);
    if (f32_nan(x))
        return ((uint64_t)(x & 0x80000000u) << 32) | 0x7ff8000000000000ull | ((uint64_t)(x & 0x003fffffu) << 29);
    return d2u((double)u2f(x));
}

static uint32_t f64_to_f32(const X64State *s, uint64_t x)
{
    x = f64_in(s, x);
    if (f64_nan(x))
        return (uint32_t)((x >> 32) & 0x80000000u) | 0x7fc00000u | (uint32_t)((x >> 29) & 0x003fffffu);
    volatile float r = (float)u2d(x);
    return f32_out(s, f2u(r));
}

// Float to integer: `truncate` for the CVTT forms, else MXCSR rounding (the host's
// rounding mode follows it during a run). Out of range or NaN: integer indefinite.
static uint64_t f_to_int(double v, bool truncate, int size)
{
    if (isnan(v)) return size == 8 ? 0x8000000000000000ull : 0x80000000ull;
    double r = truncate ? trunc(v) : nearbyint(v);
    if (size == 8) {
        if (r < -9223372036854775808.0 || r >= 9223372036854775808.0) return 0x8000000000000000ull;
        return (uint64_t)(int64_t)r;
    }
    if (r < -2147483648.0 || r > 2147483647.0) return 0x80000000ull;
    return (uint32_t)(int32_t)r;
}

static bool fcmp(int predicate, double a, double b)
{
    bool unordered = isnan(a) || isnan(b);
    switch (predicate & 7) {
    case 0: return !unordered && a == b;
    case 1: return !unordered && a < b;
    case 2: return !unordered && a <= b;
    case 3: return unordered;
    case 4: return unordered || a != b;
    case 5: return unordered || !(a < b);
    case 6: return unordered || !(a <= b);
    default: return !unordered;
    }
}

static void comis(X64State *s, double a, double b)
{
    uint64_t f;
    if (isnan(a) || isnan(b)) f = FL_ZF | FL_PF | FL_CF;
    else if (a < b) f = FL_CF;
    else if (a == b) f = FL_ZF;
    else f = 0;
    set_flags(s, FL_ARITH, f);
}

static int host_rounding(uint32_t mxcsr)
{
    switch ((mxcsr >> 13) & 3) {
    case 1: return FE_DOWNWARD;
    case 2: return FE_UPWARD;
    case 3: return FE_TOWARDZERO;
    default: return FE_TONEAREST;
    }
}

static void set_mxcsr(X64State *s, uint32_t v)
{
    s->mxcsr = v & 0xffffu;
    fesetround(host_rounding(s->mxcsr));
}

// ---------------------------------------------------------------------------
// Packed integers (SSE2)

static inline uint8_t sat_u8(int v) { return v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v; }
static inline uint8_t sat_s8(int v) { return (uint8_t)(int8_t)(v < -128 ? -128 : v > 127 ? 127 : v); }
static inline uint16_t sat_u16(int v) { return v < 0 ? 0 : v > 65535 ? 65535 : (uint16_t)v; }
static inline uint16_t sat_s16(int v) { return (uint16_t)(int16_t)(v < -32768 ? -32768 : v > 32767 ? 32767 : v); }

// The SSE2 integer operations of the 0F 60-6F, 0F 74-76 and 0F D1-FE rows
// (with a 66 prefix). Returns false for an opcode it does not have.
static bool pint(uint8_t op, X64Vec *a, const X64Vec *b)
{
    X64Vec r = {{0}};
    int i;
    uint64_t count = b->q[0];
    switch (op) {
    case 0x60: for (i = 0; i < 8; i++) { r.b[2 * i] = a->b[i]; r.b[2 * i + 1] = b->b[i]; } break;
    case 0x61: for (i = 0; i < 4; i++) { r.w[2 * i] = a->w[i]; r.w[2 * i + 1] = b->w[i]; } break;
    case 0x62: for (i = 0; i < 2; i++) { r.d[2 * i] = a->d[i]; r.d[2 * i + 1] = b->d[i]; } break;
    case 0x63: for (i = 0; i < 8; i++) { r.b[i] = sat_s8((int16_t)a->w[i]); r.b[i + 8] = sat_s8((int16_t)b->w[i]); } break;
    case 0x64: for (i = 0; i < 16; i++) r.b[i] = (int8_t)a->b[i] > (int8_t)b->b[i] ? 0xff : 0; break;
    case 0x65: for (i = 0; i < 8; i++) r.w[i] = (int16_t)a->w[i] > (int16_t)b->w[i] ? 0xffff : 0; break;
    case 0x66: for (i = 0; i < 4; i++) r.d[i] = (int32_t)a->d[i] > (int32_t)b->d[i] ? 0xffffffffu : 0; break;
    case 0x67: for (i = 0; i < 8; i++) { r.b[i] = sat_u8((int16_t)a->w[i]); r.b[i + 8] = sat_u8((int16_t)b->w[i]); } break;
    case 0x68: for (i = 0; i < 8; i++) { r.b[2 * i] = a->b[i + 8]; r.b[2 * i + 1] = b->b[i + 8]; } break;
    case 0x69: for (i = 0; i < 4; i++) { r.w[2 * i] = a->w[i + 4]; r.w[2 * i + 1] = b->w[i + 4]; } break;
    case 0x6A: for (i = 0; i < 2; i++) { r.d[2 * i] = a->d[i + 2]; r.d[2 * i + 1] = b->d[i + 2]; } break;
    case 0x6B: for (i = 0; i < 4; i++) { r.w[i] = sat_s16((int32_t)a->d[i]); r.w[i + 4] = sat_s16((int32_t)b->d[i]); } break;
    case 0x6C: r.q[0] = a->q[0]; r.q[1] = b->q[0]; break;
    case 0x6D: r.q[0] = a->q[1]; r.q[1] = b->q[1]; break;
    case 0x74: for (i = 0; i < 16; i++) r.b[i] = a->b[i] == b->b[i] ? 0xff : 0; break;
    case 0x75: for (i = 0; i < 8; i++) r.w[i] = a->w[i] == b->w[i] ? 0xffff : 0; break;
    case 0x76: for (i = 0; i < 4; i++) r.d[i] = a->d[i] == b->d[i] ? 0xffffffffu : 0; break;
    case 0xD1: for (i = 0; i < 8; i++) r.w[i] = count > 15 ? 0 : (uint16_t)(a->w[i] >> count); break;
    case 0xD2: for (i = 0; i < 4; i++) r.d[i] = count > 31 ? 0 : a->d[i] >> count; break;
    case 0xD3: for (i = 0; i < 2; i++) r.q[i] = count > 63 ? 0 : a->q[i] >> count; break;
    case 0xD4: for (i = 0; i < 2; i++) r.q[i] = a->q[i] + b->q[i]; break;
    case 0xD5: for (i = 0; i < 8; i++) r.w[i] = (uint16_t)((int32_t)(int16_t)a->w[i] * (int16_t)b->w[i]); break;
    case 0xD8: for (i = 0; i < 16; i++) r.b[i] = sat_u8(a->b[i] - b->b[i]); break;
    case 0xD9: for (i = 0; i < 8; i++) r.w[i] = sat_u16(a->w[i] - b->w[i]); break;
    case 0xDA: for (i = 0; i < 16; i++) r.b[i] = a->b[i] < b->b[i] ? a->b[i] : b->b[i]; break;
    case 0xDB: for (i = 0; i < 2; i++) r.q[i] = a->q[i] & b->q[i]; break;
    case 0xDC: for (i = 0; i < 16; i++) r.b[i] = sat_u8(a->b[i] + b->b[i]); break;
    case 0xDD: for (i = 0; i < 8; i++) r.w[i] = sat_u16(a->w[i] + b->w[i]); break;
    case 0xDE: for (i = 0; i < 16; i++) r.b[i] = a->b[i] > b->b[i] ? a->b[i] : b->b[i]; break;
    case 0xDF: for (i = 0; i < 2; i++) r.q[i] = ~a->q[i] & b->q[i]; break;
    case 0xE0: for (i = 0; i < 16; i++) r.b[i] = (uint8_t)((a->b[i] + b->b[i] + 1) >> 1); break;
    case 0xE1: for (i = 0; i < 8; i++) r.w[i] = (uint16_t)((int16_t)a->w[i] >> (count > 15 ? 15 : count)); break;
    case 0xE2: for (i = 0; i < 4; i++) r.d[i] = (uint32_t)((int32_t)a->d[i] >> (count > 31 ? 31 : count)); break;
    case 0xE3: for (i = 0; i < 8; i++) r.w[i] = (uint16_t)((a->w[i] + b->w[i] + 1) >> 1); break;
    case 0xE4: for (i = 0; i < 8; i++) r.w[i] = (uint16_t)(((uint32_t)a->w[i] * b->w[i]) >> 16); break;
    case 0xE5: for (i = 0; i < 8; i++) r.w[i] = (uint16_t)(((int32_t)(int16_t)a->w[i] * (int16_t)b->w[i]) >> 16); break;
    case 0xE8: for (i = 0; i < 16; i++) r.b[i] = sat_s8((int8_t)a->b[i] - (int8_t)b->b[i]); break;
    case 0xE9: for (i = 0; i < 8; i++) r.w[i] = sat_s16((int16_t)a->w[i] - (int16_t)b->w[i]); break;
    case 0xEA: for (i = 0; i < 8; i++) r.w[i] = (int16_t)a->w[i] < (int16_t)b->w[i] ? a->w[i] : b->w[i]; break;
    case 0xEB: for (i = 0; i < 2; i++) r.q[i] = a->q[i] | b->q[i]; break;
    case 0xEC: for (i = 0; i < 16; i++) r.b[i] = sat_s8((int8_t)a->b[i] + (int8_t)b->b[i]); break;
    case 0xED: for (i = 0; i < 8; i++) r.w[i] = sat_s16((int16_t)a->w[i] + (int16_t)b->w[i]); break;
    case 0xEE: for (i = 0; i < 8; i++) r.w[i] = (int16_t)a->w[i] > (int16_t)b->w[i] ? a->w[i] : b->w[i]; break;
    case 0xEF: for (i = 0; i < 2; i++) r.q[i] = a->q[i] ^ b->q[i]; break;
    case 0xF1: for (i = 0; i < 8; i++) r.w[i] = count > 15 ? 0 : (uint16_t)(a->w[i] << count); break;
    case 0xF2: for (i = 0; i < 4; i++) r.d[i] = count > 31 ? 0 : a->d[i] << count; break;
    case 0xF3: for (i = 0; i < 2; i++) r.q[i] = count > 63 ? 0 : a->q[i] << count; break;
    case 0xF4: for (i = 0; i < 2; i++) r.q[i] = (uint64_t)a->d[2 * i] * b->d[2 * i]; break;
    case 0xF5:
        for (i = 0; i < 4; i++)
            r.d[i] = (uint32_t)((int64_t)(int16_t)a->w[2 * i] * (int16_t)b->w[2 * i]
                                + (int64_t)(int16_t)a->w[2 * i + 1] * (int16_t)b->w[2 * i + 1]);
        break;
    case 0xF6:
        for (i = 0; i < 2; i++) {
            unsigned sum = 0;
            for (int j = 0; j < 8; j++) sum += (unsigned)abs(a->b[8 * i + j] - b->b[8 * i + j]);
            r.q[i] = sum;
        }
        break;
    case 0xF8: for (i = 0; i < 16; i++) r.b[i] = (uint8_t)(a->b[i] - b->b[i]); break;
    case 0xF9: for (i = 0; i < 8; i++) r.w[i] = (uint16_t)(a->w[i] - b->w[i]); break;
    case 0xFA: for (i = 0; i < 4; i++) r.d[i] = a->d[i] - b->d[i]; break;
    case 0xFB: for (i = 0; i < 2; i++) r.q[i] = a->q[i] - b->q[i]; break;
    case 0xFC: for (i = 0; i < 16; i++) r.b[i] = (uint8_t)(a->b[i] + b->b[i]); break;
    case 0xFD: for (i = 0; i < 8; i++) r.w[i] = (uint16_t)(a->w[i] + b->w[i]); break;
    case 0xFE: for (i = 0; i < 4; i++) r.d[i] = a->d[i] + b->d[i]; break;
    default: return false;
    }
    *a = r;
    return true;
}

// ---------------------------------------------------------------------------
// SSE instructions (two-byte map, and the 0F 38 / 0F 3A forms implemented)

static inline X64Vec xm(const X64State *s, const Dec *d, int bytes)
{
    X64Vec v;
    if (!d->mem) return s->xmm[d->rm];
    vec_read(&v, d->ea, bytes);
    return v;
}

static int sse(X64State *s, Dec *d, int map, uint8_t op)
{
    // The mandatory prefix: 0 none, 1 66, 2 F3, 3 F2 (F2/F3 win over 66).
    int pfx = d->rep == 0xF3 ? 2 : d->rep == 0xF2 ? 3 : d->opsize ? 1 : 0;
    int i;
    uint8_t imm = 0;

    if (map == 2) {
        if (op != 0x00 || pfx != 1) return X64_EXIT_UNSUPPORTED;
        modrm(d);
        if (!finish(s, d)) return X64_EXIT_ILLEGAL;
        s->rip = d->next;
        X64Vec b = xm(s, d, 16), *a = &s->xmm[d->reg], r = {{0}};
        for (i = 0; i < 16; i++) r.b[i] = (b.b[i] & 0x80) ? 0 : a->b[b.b[i] & 15];   // PSHUFB
        *a = r;
        return CONT;
    }
    if (map == 3) {
        if (op != 0x0F || pfx != 1) return X64_EXIT_UNSUPPORTED;
        modrm(d);
        imm = fetch8(d);
        if (!finish(s, d)) return X64_EXIT_ILLEGAL;
        s->rip = d->next;
        X64Vec b = xm(s, d, 16), *a = &s->xmm[d->reg], r = {{0}};   // PALIGNR
        uint8_t cat[32];
        memcpy(cat, b.b, 16);
        memcpy(cat + 16, a->b, 16);
        for (i = 0; i < 16; i++) r.b[i] = (unsigned)(i + imm) < 32 ? cat[i + imm] : 0;
        *a = r;
        return CONT;
    }

    // Opcodes whose 66-less form is MMX (or that need a prefix to exist at all).
    bool sse_int = (op >= 0x60 && op <= 0x7F) || op >= 0xD0 || op == 0xC4 || op == 0xC5;
    if (sse_int && pfx == 0) return X64_EXIT_UNSUPPORTED;

    modrm(d);
    if (op == 0x70 || op == 0x71 || op == 0x72 || op == 0x73 || op == 0xC2 || op == 0xC4 || op == 0xC5 || op == 0xC6)
        imm = fetch8(d);
    if (!finish(s, d)) return X64_EXIT_ILLEGAL;
    s->rip = d->next;

    X64Vec *a = &s->xmm[d->reg];
    X64Vec b = {{0}}, r = {{0}};

    switch (op) {
    case 0x10: // MOVUPS MOVUPD MOVSS MOVSD (load)
        if (pfx == 2) { b = xm(s, d, 4); if (d->mem) a->q[0] = a->q[1] = 0; a->d[0] = b.d[0]; }
        else if (pfx == 3) { b = xm(s, d, 8); if (d->mem) a->q[1] = 0; a->q[0] = b.q[0]; }
        else *a = xm(s, d, 16);
        return CONT;
    case 0x11: // the stores
        if (pfx == 2) { if (d->mem) vec_write(d->ea, a, 4); else s->xmm[d->rm].d[0] = a->d[0]; }
        else if (pfx == 3) { if (d->mem) vec_write(d->ea, a, 8); else s->xmm[d->rm].q[0] = a->q[0]; }
        else if (d->mem) vec_write(d->ea, a, 16);
        else s->xmm[d->rm] = *a;
        return CONT;
    case 0x12:
        if (pfx == 2) { b = xm(s, d, 16); r.d[0] = r.d[1] = b.d[0]; r.d[2] = r.d[3] = b.d[2]; *a = r; }    // MOVSLDUP
        else if (pfx == 3) { b = xm(s, d, 8); a->q[0] = a->q[1] = b.q[0]; }                                 // MOVDDUP
        else if (d->mem) { b = xm(s, d, 8); a->q[0] = b.q[0]; }                                              // MOVLPS/MOVLPD
        else if (pfx == 0) a->q[0] = s->xmm[d->rm].q[1];                                                     // MOVHLPS
        else return X64_EXIT_ILLEGAL;
        return CONT;
    case 0x13: case 0x17: // MOVLPS/MOVLPD, MOVHPS/MOVHPD stores
        if (!d->mem || pfx > 1) return X64_EXIT_ILLEGAL;
        b.q[0] = a->q[op == 0x17];
        vec_write(d->ea, &b, 8);
        return CONT;
    case 0x14: case 0x15: // UNPCKLPS/PD, UNPCKHPS/PD
        if (pfx > 1) return X64_EXIT_ILLEGAL;
        b = xm(s, d, 16);
        if (pfx == 0) {
            int h = op == 0x15 ? 2 : 0;
            r.d[0] = a->d[h]; r.d[1] = b.d[h]; r.d[2] = a->d[h + 1]; r.d[3] = b.d[h + 1];
        } else {
            r.q[0] = a->q[op == 0x15]; r.q[1] = b.q[op == 0x15];
        }
        *a = r;
        return CONT;
    case 0x16:
        if (pfx == 2) { b = xm(s, d, 16); r.d[0] = r.d[1] = b.d[1]; r.d[2] = r.d[3] = b.d[3]; *a = r; }    // MOVSHDUP
        else if (pfx == 3) return X64_EXIT_ILLEGAL;
        else if (d->mem) { b = xm(s, d, 8); a->q[1] = b.q[0]; }                                              // MOVHPS/MOVHPD
        else if (pfx == 0) a->q[1] = s->xmm[d->rm].q[0];                                                     // MOVLHPS
        else return X64_EXIT_ILLEGAL;
        return CONT;
    case 0x28: // MOVAPS MOVAPD
        if (pfx > 1) return X64_EXIT_ILLEGAL;
        *a = xm(s, d, 16);
        return CONT;
    case 0x29: case 0x2B: // MOVAPS/MOVAPD store, MOVNTPS/MOVNTPD
        if (pfx > 1 || (op == 0x2B && !d->mem)) return X64_EXIT_ILLEGAL;
        if (d->mem) vec_write(d->ea, a, 16); else s->xmm[d->rm] = *a;
        return CONT;
    case 0x2A: { // CVTSI2SS CVTSI2SD
        if (pfx < 2) return X64_EXIT_UNSUPPORTED;   // CVTPI2PS/PD take MMX
        int size = (d->rex & 8) ? 8 : 4;
        int64_t v = sext(get_rm(s, d, size), size);
        if (pfx == 2) { volatile float f = (float)v; a->d[0] = f2u(f); }
        else { volatile double f = (double)v; a->q[0] = d2u(f); }
        return CONT;
    }
    case 0x2C: case 0x2D: { // CVTTSS2SI CVTSS2SI CVTTSD2SI CVTSD2SI
        if (pfx < 2) return X64_EXIT_UNSUPPORTED;
        int size = (d->rex & 8) ? 8 : 4;
        double v;
        if (pfx == 2) { b = xm(s, d, 4); v = u2f(f32_in(s, b.d[0])); }
        else { b = xm(s, d, 8); v = u2d(f64_in(s, b.q[0])); }
        set_reg(s, d, d->reg, size, f_to_int(v, op == 0x2C, size));
        return CONT;
    }
    case 0x2E: case 0x2F: // UCOMISS UCOMISD COMISS COMISD
        if (pfx > 1) return X64_EXIT_ILLEGAL;
        if (pfx == 0) { b = xm(s, d, 4); comis(s, u2f(f32_in(s, a->d[0])), u2f(f32_in(s, b.d[0]))); }
        else { b = xm(s, d, 8); comis(s, u2d(f64_in(s, a->q[0])), u2d(f64_in(s, b.q[0]))); }
        return CONT;
    case 0x50: { // MOVMSKPS MOVMSKPD
        if (pfx > 1 || d->mem) return X64_EXIT_ILLEGAL;
        const X64Vec *v = &s->xmm[d->rm];
        uint64_t m = 0;
        if (pfx == 0) for (i = 0; i < 4; i++) m |= (uint64_t)(v->d[i] >> 31) << i;
        else for (i = 0; i < 2; i++) m |= (v->q[i] >> 63) << i;
        s->gpr[d->reg] = m;
        return CONT;
    }
    case 0x51: case 0x52: case 0x53: case 0x58: case 0x59: case 0x5C: case 0x5D: case 0x5E: case 0x5F: {
        int fop;
        switch (op) {
        case 0x51: fop = FOP_SQRT; break;
        case 0x52: fop = FOP_RSQRT; break;
        case 0x53: fop = FOP_RCP; break;
        default: fop = op - 0x58; break;  // ADD MUL (5A, 5B elsewhere) SUB MIN DIV MAX
        }
        if ((fop == FOP_RSQRT || fop == FOP_RCP) && (pfx & 1)) return X64_EXIT_ILLEGAL;
        switch (pfx) {
        case 0: b = xm(s, d, 16); for (i = 0; i < 4; i++) a->d[i] = f32_op(s, fop, a->d[i], b.d[i]); break;
        case 1: b = xm(s, d, 16); for (i = 0; i < 2; i++) a->q[i] = f64_op(s, fop, a->q[i], b.q[i]); break;
        case 2: b = xm(s, d, 4); a->d[0] = f32_op(s, fop, a->d[0], b.d[0]); break;
        default: b = xm(s, d, 8); a->q[0] = f64_op(s, fop, a->q[0], b.q[0]); break;
        }
        return CONT;
    }
    case 0x54: case 0x55: case 0x56: case 0x57: // ANDPS ANDNPS ORPS XORPS (and PD)
        if (pfx > 1) return X64_EXIT_ILLEGAL;
        b = xm(s, d, 16);
        for (i = 0; i < 2; i++) {
            if (op == 0x54) a->q[i] &= b.q[i];
            else if (op == 0x55) a->q[i] = ~a->q[i] & b.q[i];
            else if (op == 0x56) a->q[i] |= b.q[i];
            else a->q[i] ^= b.q[i];
        }
        return CONT;
    case 0x5A: // CVTPS2PD CVTPD2PS CVTSS2SD CVTSD2SS
        switch (pfx) {
        case 0: b = xm(s, d, 8); r.q[0] = f32_to_f64(s, b.d[0]); r.q[1] = f32_to_f64(s, b.d[1]); *a = r; break;
        case 1: b = xm(s, d, 16); r.d[0] = f64_to_f32(s, b.q[0]); r.d[1] = f64_to_f32(s, b.q[1]); r.q[1] = 0; *a = r; break;
        case 2: b = xm(s, d, 4); a->q[0] = f32_to_f64(s, b.d[0]); break;
        default: b = xm(s, d, 8); a->d[0] = f64_to_f32(s, b.q[0]); break;
        }
        return CONT;
    case 0x5B: // CVTDQ2PS CVTPS2DQ CVTTPS2DQ
        if (pfx == 3) return X64_EXIT_ILLEGAL;
        b = xm(s, d, 16);
        for (i = 0; i < 4; i++) {
            if (pfx == 0) { volatile float f = (float)(int32_t)b.d[i]; a->d[i] = f32_out(s, f2u(f)); }
            else a->d[i] = (uint32_t)f_to_int(u2f(f32_in(s, b.d[i])), pfx == 2, 4);
        }
        return CONT;
    case 0x6E: // MOVD/MOVQ xmm, r/m
        a->q[1] = 0;
        a->q[0] = get_rm(s, d, (d->rex & 8) ? 8 : 4);
        return CONT;
    case 0x6F: // MOVDQA MOVDQU
        if (pfx == 3) return X64_EXIT_ILLEGAL;
        *a = xm(s, d, 16);
        return CONT;
    case 0x70: { // PSHUFD PSHUFHW PSHUFLW
        b = xm(s, d, 16);
        r = b;
        if (pfx == 1) for (i = 0; i < 4; i++) r.d[i] = b.d[(imm >> (2 * i)) & 3];
        else if (pfx == 2) for (i = 0; i < 4; i++) r.w[4 + i] = b.w[4 + ((imm >> (2 * i)) & 3)];
        else for (i = 0; i < 4; i++) r.w[i] = b.w[(imm >> (2 * i)) & 3];
        *a = r;
        return CONT;
    }
    case 0x71: case 0x72: case 0x73: { // shifts by an immediate
        if (pfx != 1 || d->mem) return X64_EXIT_ILLEGAL;
        X64Vec *v = &s->xmm[d->rm];
        X64Vec count = {{0}};
        count.q[0] = imm;
        uint8_t equivalent = 0;
        switch ((op << 4) | d->regf) {
        case 0x712: equivalent = 0xD1; break;
        case 0x714: equivalent = 0xE1; break;
        case 0x716: equivalent = 0xF1; break;
        case 0x722: equivalent = 0xD2; break;
        case 0x724: equivalent = 0xE2; break;
        case 0x726: equivalent = 0xF2; break;
        case 0x732: equivalent = 0xD3; break;
        case 0x736: equivalent = 0xF3; break;
        case 0x733: case 0x737: { // PSRLDQ PSLLDQ
            unsigned n = imm > 16 ? 16 : imm;
            memset(&r, 0, sizeof r);
            if (d->regf == 3) memcpy(r.b, v->b + n, 16 - n);
            else memcpy(r.b + n, v->b, 16 - n);
            *v = r;
            return CONT;
        }
        default: return X64_EXIT_ILLEGAL;
        }
        pint(equivalent, v, &count);
        return CONT;
    }
    case 0x7C: case 0x7D: { // HADDPD HADDPS HSUBPD HSUBPS
        int fop = op == 0x7C ? FOP_ADD : FOP_SUB;
        b = xm(s, d, 16);
        if (pfx == 1) { r.q[0] = f64_op(s, fop, a->q[0], a->q[1]); r.q[1] = f64_op(s, fop, b.q[0], b.q[1]); }
        else if (pfx == 3) {
            r.d[0] = f32_op(s, fop, a->d[0], a->d[1]); r.d[1] = f32_op(s, fop, a->d[2], a->d[3]);
            r.d[2] = f32_op(s, fop, b.d[0], b.d[1]); r.d[3] = f32_op(s, fop, b.d[2], b.d[3]);
        } else return X64_EXIT_ILLEGAL;
        *a = r;
        return CONT;
    }
    case 0x7E:
        if (pfx == 2) { b = xm(s, d, 8); a->q[0] = b.q[0]; a->q[1] = 0; }       // MOVQ xmm, xmm/m64
        else if (pfx == 1) set_rm(s, d, (d->rex & 8) ? 8 : 4, a->q[0]);        // MOVD/MOVQ r/m, xmm
        else return X64_EXIT_ILLEGAL;
        return CONT;
    case 0x7F: // MOVDQA MOVDQU stores
        if (pfx == 3) return X64_EXIT_ILLEGAL;
        if (d->mem) vec_write(d->ea, a, 16); else s->xmm[d->rm] = *a;
        return CONT;
    case 0xC2: { // CMPPS CMPPD CMPSS CMPSD
        switch (pfx) {
        case 0: b = xm(s, d, 16);
            for (i = 0; i < 4; i++) a->d[i] = fcmp(imm, u2f(f32_in(s, a->d[i])), u2f(f32_in(s, b.d[i]))) ? ~0u : 0;
            break;
        case 1: b = xm(s, d, 16);
            for (i = 0; i < 2; i++) a->q[i] = fcmp(imm, u2d(f64_in(s, a->q[i])), u2d(f64_in(s, b.q[i]))) ? ~0ull : 0;
            break;
        case 2: b = xm(s, d, 4); a->d[0] = fcmp(imm, u2f(f32_in(s, a->d[0])), u2f(f32_in(s, b.d[0]))) ? ~0u : 0; break;
        default: b = xm(s, d, 8); a->q[0] = fcmp(imm, u2d(f64_in(s, a->q[0])), u2d(f64_in(s, b.q[0]))) ? ~0ull : 0; break;
        }
        return CONT;
    }
    case 0xC4: // PINSRW
        if (pfx != 1) return X64_EXIT_ILLEGAL;
        a->w[imm & 7] = (uint16_t)get_rm(s, d, 2);
        return CONT;
    case 0xC5: // PEXTRW
        if (pfx != 1 || d->mem) return X64_EXIT_ILLEGAL;
        s->gpr[d->reg] = s->xmm[d->rm].w[imm & 7];
        return CONT;
    case 0xC6: // SHUFPS SHUFPD
        if (pfx > 1) return X64_EXIT_ILLEGAL;
        b = xm(s, d, 16);
        if (pfx == 0) {
            r.d[0] = a->d[imm & 3]; r.d[1] = a->d[(imm >> 2) & 3];
            r.d[2] = b.d[(imm >> 4) & 3]; r.d[3] = b.d[(imm >> 6) & 3];
        } else {
            r.q[0] = a->q[imm & 1]; r.q[1] = b.q[(imm >> 1) & 1];
        }
        *a = r;
        return CONT;
    case 0xD0: // ADDSUBPD ADDSUBPS
        b = xm(s, d, 16);
        if (pfx == 1) { a->q[0] = f64_op(s, FOP_SUB, a->q[0], b.q[0]); a->q[1] = f64_op(s, FOP_ADD, a->q[1], b.q[1]); }
        else if (pfx == 3) for (i = 0; i < 4; i++) a->d[i] = f32_op(s, (i & 1) ? FOP_ADD : FOP_SUB, a->d[i], b.d[i]);
        else return X64_EXIT_ILLEGAL;
        return CONT;
    case 0xD6: // MOVQ xmm/m64, xmm
        if (pfx != 1) return X64_EXIT_UNSUPPORTED;   // MOVQ2DQ/MOVDQ2Q are MMX
        if (d->mem) vec_write(d->ea, a, 8);
        else { s->xmm[d->rm].q[0] = a->q[0]; s->xmm[d->rm].q[1] = 0; }
        return CONT;
    case 0xD7: { // PMOVMSKB
        if (pfx != 1 || d->mem) return X64_EXIT_ILLEGAL;
        uint64_t m = 0;
        for (i = 0; i < 16; i++) m |= (uint64_t)(s->xmm[d->rm].b[i] >> 7) << i;
        s->gpr[d->reg] = m;
        return CONT;
    }
    case 0xE6: // CVTTPD2DQ CVTDQ2PD CVTPD2DQ
        if (pfx == 2) {
            b = xm(s, d, 8);
            r.q[0] = d2u((double)(int32_t)b.d[0]);
            r.q[1] = d2u((double)(int32_t)b.d[1]);
        } else {
            b = xm(s, d, 16);
            for (i = 0; i < 2; i++) r.d[i] = (uint32_t)f_to_int(u2d(f64_in(s, b.q[i])), pfx == 1, 4);
            r.q[1] = 0;
        }
        *a = r;
        return CONT;
    case 0xE7: // MOVNTDQ
        if (pfx != 1 || !d->mem) return X64_EXIT_ILLEGAL;
        vec_write(d->ea, a, 16);
        return CONT;
    case 0xF0: // LDDQU
        if (pfx != 3 || !d->mem) return X64_EXIT_ILLEGAL;
        *a = xm(s, d, 16);
        return CONT;
    case 0xF7: { // MASKMOVDQU
        if (pfx != 1 || d->mem) return X64_EXIT_ILLEGAL;
        uint64_t base = (d->addrsize ? s->gpr[X64_RDI] & 0xffffffffull : s->gpr[X64_RDI]) + seg_base(s, d);
        for (i = 0; i < 16; i++)
            if (s->xmm[d->rm].b[i] & 0x80) mem_write(base + (uint64_t)i, 1, a->b[i]);
        return CONT;
    }
    }
    if (sse_int && pfx == 1) {
        b = xm(s, d, 16);
        if (pint(op, a, &b)) return CONT;
    }
    return X64_EXIT_UNSUPPORTED;
}

// ---------------------------------------------------------------------------
// String instructions

static int string_op(X64State *s, Dec *d, uint8_t op)
{
    int size = (op & 1) ? osz(d) : 1;
    bool a32 = d->addrsize;
    uint64_t amask = a32 ? 0xffffffffull : ~0ull;
    int asize = a32 ? 4 : 8;
    int64_t step = (s->rflags & FL_DF) ? -size : size;
    uint64_t src_base = seg_base(s, d);
    bool rep = d->rep != 0;
    uint64_t count = rep ? s->gpr[X64_RCX] & amask : 1;
    uint64_t rsi = s->gpr[X64_RSI] & amask, rdi = s->gpr[X64_RDI] & amask;
    uint64_t done = 0;

    switch (op) {
    case 0xA4: case 0xA5: { // MOVS
        uint64_t bytes = count * (uint64_t)size;
        uint64_t src = rsi + src_base, dst = rdi;
        if (rep && step > 0 && !a32 && count && (dst <= src || dst >= src + bytes)) {
#if X64_INTERP_TSO
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
            memmove((void *)(uintptr_t)dst, (const void *)(uintptr_t)src, (size_t)bytes);
#if X64_INTERP_TSO
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
            rsi += bytes; rdi += bytes; done = count;
        } else {
            for (; done < count; done++) {
                mem_write(rdi, size, mem_read((rsi & amask) + src_base, size));
                rsi = (rsi + (uint64_t)step) & amask;
                rdi = (rdi + (uint64_t)step) & amask;
            }
        }
        break;
    }
    case 0xAA: case 0xAB: { // STOS
        uint64_t v = s->gpr[X64_RAX] & mask_of(size);
        if (rep && step > 0 && !a32 && size == 1 && count) {
            memset((void *)(uintptr_t)rdi, (int)v, (size_t)count);
            rdi += count; done = count;
        } else {
            for (; done < count; done++) {
                mem_write(rdi, size, v);
                rdi = (rdi + (uint64_t)step) & amask;
            }
        }
        break;
    }
    case 0xAC: case 0xAD: // LODS
        for (; done < count; done++) {
            set_reg(s, d, X64_RAX, size, mem_read(rsi + src_base, size));
            rsi = (rsi + (uint64_t)step) & amask;
        }
        break;
    case 0xA6: case 0xA7: case 0xAE: case 0xAF: // CMPS, SCAS
        while (done < count) {
            uint64_t a, b;
            if (op <= 0xA7) {
                a = mem_read(rsi + src_base, size);
                b = mem_read(rdi, size);
                rsi = (rsi + (uint64_t)step) & amask;
            } else {
                a = s->gpr[X64_RAX];
                b = mem_read(rdi, size);
            }
            rdi = (rdi + (uint64_t)step) & amask;
            op_sub(s, a, b, 0, size);
            done++;
            if (d->rep == 0xF3 && !(s->rflags & FL_ZF)) break;
            if (d->rep == 0xF2 && (s->rflags & FL_ZF)) break;
        }
        break;
    }
    if (op <= 0xA7 || op == 0xAC || op == 0xAD) set_reg(s, d, X64_RSI, asize, rsi);   // MOVS CMPS LODS
    if (op != 0xAC && op != 0xAD) set_reg(s, d, X64_RDI, asize, rdi);                 // all but LODS
    if (rep) set_reg(s, d, X64_RCX, asize, count - done);
    return CONT;
}

// ---------------------------------------------------------------------------
// One instruction

static inline bool native(const X64Env *env, uint64_t target)
{
    return env->is_native && env->is_native(env->user, target);
}

static int jump(X64State *s, const X64Env *env, uint64_t target)
{
    s->rip = target;
    return native(env, target) ? X64_EXIT_NATIVE : CONT;
}

static int call(X64State *s, const X64Env *env, uint64_t target, uint64_t ret)
{
    push(s, ret, 8);
    return jump(s, env, target);
}

static uint64_t rdtsc(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// FXSAVE's 512-byte area: control and status words, MXCSR, the x87 registers
// (zero: no x87 arithmetic here) and the XMM registers. Bytes 416-511 are left
// alone, as x86 leaves 464-511 to software.
static void fxsave(const X64State *s, uint64_t addr)
{
    uint8_t area[416];
    memset(area, 0, sizeof area);
    memcpy(area + 0, &s->fcw, 2);
    memcpy(area + 2, &s->fsw, 2);
    memcpy(area + 24, &s->mxcsr, 4);
    uint32_t mask = 0xffff;
    memcpy(area + 28, &mask, 4);
    memcpy(area + 160, s->xmm, sizeof s->xmm);
    memcpy((void *)(uintptr_t)addr, area, sizeof area);
}

static void fxrstor(X64State *s, uint64_t addr)
{
    const uint8_t *area = (const uint8_t *)(uintptr_t)addr;
    uint32_t mxcsr;
    memcpy(&s->fcw, area + 0, 2);
    memcpy(&s->fsw, area + 2, 2);
    memcpy(&mxcsr, area + 24, 4);
    memcpy(s->xmm, area + 160, sizeof s->xmm);
    set_mxcsr(s, mxcsr);
}

// The opcodes LOCK may precede (with a memory destination, checked where decoded).
static bool lockable(int map, uint8_t op)
{
    if (map == 0) {
        if (op < 0x40 && (op & 7) < 2 && (op >> 3) != ALU_CMP) return true;
        return op == 0x80 || op == 0x81 || op == 0x83 || op == 0x86 || op == 0x87 ||
               op == 0xF6 || op == 0xF7 || op == 0xFE || op == 0xFF;
    }
    return map == 1 && (op == 0xAB || op == 0xB3 || op == 0xBB || op == 0xBA || op == 0xB0 ||
                        op == 0xB1 || op == 0xC0 || op == 0xC1 || op == 0xC7);
}

#define FIN() do { if (!finish(s, d)) return X64_EXIT_ILLEGAL; s->rip = d->next; } while (0)
#define NEED_MEM() do { if (!d->mem) return X64_EXIT_ILLEGAL; } while (0)
#define LOCK_MEM() do { if (d->lock && !d->mem) return X64_EXIT_ILLEGAL; } while (0)

static int step(X64State *s, const X64Env *env, Dec *d)
{
    uint8_t op;
    int map = 0, size;
    uint64_t v = 0, t;

    d->start = s->rip;
    d->begin = d->p = (const uint8_t *)(uintptr_t)s->rip;
    d->rex = 0; d->rep = 0; d->opsize = d->addrsize = d->lock = false; d->seg = -1; d->mem = false;

    for (;;) {
        if (d->p - d->begin >= 15) return X64_EXIT_ILLEGAL;
        op = *d->p;
        if ((op & 0xF0) == 0x40) { d->rex = op; d->p++; continue; }
        switch (op) {
        case 0x66: d->opsize = true; break;
        case 0x67: d->addrsize = true; break;
        case 0xF2: case 0xF3: d->rep = op; break;
        case 0xF0: d->lock = true; break;
        case 0x26: case 0x2E: case 0x36: case 0x3E: break;
        case 0x64: d->seg = 4; break;
        case 0x65: d->seg = 5; break;
        default: goto opcode;
        }
        d->rex = 0;     // a REX prefix only counts right before the opcode
        d->p++;
    }
opcode:
    d->p++;
    if (op == 0x0F) {
        op = fetch8(d);
        map = 1;
        if (op == 0x38 || op == 0x3A) { map = op == 0x38 ? 2 : 3; op = fetch8(d); }
    }
    if (d->lock && !lockable(map, op)) return X64_EXIT_ILLEGAL;

    if (map == 0) switch (op) {
    case 0x00: case 0x01: case 0x02: case 0x03: case 0x04: case 0x05:
    case 0x08: case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D:
    case 0x10: case 0x11: case 0x12: case 0x13: case 0x14: case 0x15:
    case 0x18: case 0x19: case 0x1A: case 0x1B: case 0x1C: case 0x1D:
    case 0x20: case 0x21: case 0x22: case 0x23: case 0x24: case 0x25:
    case 0x28: case 0x29: case 0x2A: case 0x2B: case 0x2C: case 0x2D:
    case 0x30: case 0x31: case 0x32: case 0x33: case 0x34: case 0x35:
    case 0x38: case 0x39: case 0x3A: case 0x3B: case 0x3C: case 0x3D: {
        int alu_op = op >> 3, form = op & 7;
        size = (form & 1) ? osz(d) : 1;
        if (form >= 4) {
            v = size == 1 ? fetch8(d) : size == 2 ? fetch16(d) : (uint64_t)(int64_t)(int32_t)fetch32(d);
            FIN();
            uint64_t r = alu(s, alu_op, get_reg(s, d, X64_RAX, size), v, s->rflags & FL_CF, size);
            if (alu_op != ALU_CMP) set_reg(s, d, X64_RAX, size, r);
            return CONT;
        }
        modrm(d);
        FIN();
        if (form < 2) {
            LOCK_MEM();
            if (alu_op == ALU_CMP) alu(s, alu_op, get_rm(s, d, size), get_reg(s, d, d->reg, size), 0, size);
            else rmw(s, d, size, RMW_ALU, alu_op, get_reg(s, d, d->reg, size));
        } else {
            uint64_t r = alu(s, alu_op, get_reg(s, d, d->reg, size), get_rm(s, d, size), s->rflags & FL_CF, size);
            if (alu_op != ALU_CMP) set_reg(s, d, d->reg, size, r);
        }
        return CONT;
    }
    case 0x50: case 0x51: case 0x52: case 0x53: case 0x54: case 0x55: case 0x56: case 0x57:
        FIN();
        size = d->opsize ? 2 : 8;
        push(s, s->gpr[(op & 7) | ((d->rex & 1) << 3)] & mask_of(size), size);
        return CONT;
    case 0x58: case 0x59: case 0x5A: case 0x5B: case 0x5C: case 0x5D: case 0x5E: case 0x5F:
        FIN();
        size = d->opsize ? 2 : 8;
        v = pop(s, size);
        set_reg(s, d, (op & 7) | ((d->rex & 1) << 3), size, v);
        return CONT;
    case 0x63: // MOVSXD
        size = osz(d);
        modrm(d);
        FIN();
        if (size == 8) s->gpr[d->reg] = (uint64_t)(int64_t)(int32_t)get_rm(s, d, 4);
        else set_reg(s, d, d->reg, size, get_rm(s, d, size));
        return CONT;
    case 0x68: case 0x6A: // PUSH imm
        size = d->opsize ? 2 : 8;
        v = op == 0x6A ? (uint64_t)(int64_t)(int8_t)fetch8(d)
          : size == 2 ? fetch16(d) : (uint64_t)(int64_t)(int32_t)fetch32(d);
        FIN();
        push(s, v & mask_of(size), size);
        return CONT;
    case 0x69: case 0x6B: { // IMUL Gv, Ev, imm
        size = osz(d);
        modrm(d);
        int64_t imm = op == 0x6B ? (int8_t)fetch8(d) : size == 2 ? (int16_t)fetch16(d) : (int32_t)fetch32(d);
        FIN();
        int64_t a = sext(get_rm(s, d, size), size);
        __int128 full = (__int128)a * imm;
        uint64_t r = (uint64_t)full & mask_of(size);
        bool over = (__int128)sext(r, size) != full;
        set_flags(s, FL_ARITH, (over ? FL_CF | FL_OF : 0) | szp(r, size));
        set_reg(s, d, d->reg, size, r);
        return CONT;
    }
    case 0x70: case 0x71: case 0x72: case 0x73: case 0x74: case 0x75: case 0x76: case 0x77:
    case 0x78: case 0x79: case 0x7A: case 0x7B: case 0x7C: case 0x7D: case 0x7E: case 0x7F:
        t = (uint64_t)(int64_t)(int8_t)fetch8(d);
        FIN();
        if (cond(s, op & 15)) s->rip = d->next + t;
        return CONT;
    case 0x80: case 0x81: case 0x83: {
        size = op == 0x80 ? 1 : osz(d);
        modrm(d);
        v = (op == 0x81 && size != 2) ? (uint64_t)(int64_t)(int32_t)fetch32(d)
          : (op == 0x81) ? fetch16(d) : (uint64_t)(int64_t)(int8_t)fetch8(d);
        FIN();
        if (d->regf == ALU_CMP) {
            if (d->lock) return X64_EXIT_ILLEGAL;
            alu(s, ALU_CMP, get_rm(s, d, size), v, 0, size);
        } else {
            LOCK_MEM();
            rmw(s, d, size, RMW_ALU, d->regf, v);
        }
        return CONT;
    }
    case 0x84: case 0x85: // TEST
        size = op == 0x84 ? 1 : osz(d);
        modrm(d);
        FIN();
        alu(s, ALU_AND, get_rm(s, d, size), get_reg(s, d, d->reg, size), 0, size);
        return CONT;
    case 0x86: case 0x87: // XCHG
        size = op == 0x86 ? 1 : osz(d);
        modrm(d);
        FIN();
        LOCK_MEM();
        if (d->mem) v = mem_xchg(d->ea, size, get_reg(s, d, d->reg, size));
        else { v = get_rm(s, d, size); set_rm(s, d, size, get_reg(s, d, d->reg, size)); }
        set_reg(s, d, d->reg, size, v);
        return CONT;
    case 0x88: case 0x89: case 0x8A: case 0x8B: // MOV
        size = (op & 1) ? osz(d) : 1;
        modrm(d);
        FIN();
        if (op < 0x8A) set_rm(s, d, size, get_reg(s, d, d->reg, size));
        else set_reg(s, d, d->reg, size, get_rm(s, d, size));
        return CONT;
    case 0x8C: { // MOV r/m, Sreg
        modrm(d);
        FIN();
        uint16_t sel;
        switch (d->regf) {
        case 0: sel = s->seg_es; break;
        case 1: sel = s->seg_cs; break;
        case 2: sel = s->seg_ss; break;
        case 3: sel = s->seg_ds; break;
        case 4: sel = s->seg_fs; break;
        case 5: sel = s->seg_gs; break;
        default: return X64_EXIT_ILLEGAL;
        }
        if (d->mem) mem_write(d->ea, 2, sel); else set_reg(s, d, d->rm, osz(d), sel);
        return CONT;
    }
    case 0x8D: // LEA
        size = osz(d);
        modrm(d);
        FIN();
        if (!d->mem) return X64_EXIT_ILLEGAL;
        set_reg(s, d, d->reg, size, d->off);
        return CONT;
    case 0x8F: // POP Ev (8F with reg != 0 is XOP)
        modrm(d);
        if (d->regf != 0) return X64_EXIT_UNSUPPORTED;
        FIN();
        size = d->opsize ? 2 : 8;
        v = pop(s, size);
        if (d->mem && d->base == X64_RSP) d->ea += (uint64_t)size;   // the address uses the new RSP
        set_rm(s, d, size, v);
        return CONT;
    case 0x90: case 0x91: case 0x92: case 0x93: case 0x94: case 0x95: case 0x96: case 0x97: {
        FIN();
        int r = (op & 7) | ((d->rex & 1) << 3);
        if (r == X64_RAX) {
            if (op == 0x90 && !(d->rex & 8) && !d->opsize) return CONT;   // NOP, PAUSE
            size = osz(d);
            set_reg(s, d, X64_RAX, size, s->gpr[X64_RAX]);               // xchg eax, eax zero-extends
            return CONT;
        }
        size = osz(d);
        v = get_reg(s, d, r, size);
        set_reg(s, d, r, size, get_reg(s, d, X64_RAX, size));
        set_reg(s, d, X64_RAX, size, v);
        return CONT;
    }
    case 0x98: // CBW CWDE CDQE
        FIN();
        size = osz(d);
        set_reg(s, d, X64_RAX, size, (uint64_t)sext(s->gpr[X64_RAX], size / 2));
        return CONT;
    case 0x99: // CWD CDQ CQO
        FIN();
        size = osz(d);
        set_reg(s, d, X64_RDX, size, (s->gpr[X64_RAX] & sign_of(size)) ? ~0ull : 0);
        return CONT;
    case 0x9B: // FWAIT
        FIN();
        return CONT;
    case 0x9C: // PUSHF
        FIN();
        size = d->opsize ? 2 : 8;
        push(s, (s->rflags | 2) & mask_of(size) & ~0x30000ull, size);
        return CONT;
    case 0x9D: // POPF
        FIN();
        size = d->opsize ? 2 : 8;
        v = pop(s, size);
        set_flags(s, FL_POPF & mask_of(size), v);
        s->rflags |= 2 | FL_IF;
        return CONT;
    case 0x9E: // SAHF
        FIN();
        set_flags(s, FL_CF | FL_PF | FL_AF | FL_ZF | FL_SF, s->gpr[X64_RAX] >> 8);
        return CONT;
    case 0x9F: // LAHF (always AH, REX or not)
        FIN();
        s->gpr[X64_RAX] = (s->gpr[X64_RAX] & ~0xff00ull)
                        | (((s->rflags & (FL_CF | FL_PF | FL_AF | FL_ZF | FL_SF)) | 2) << 8);
        return CONT;
    case 0xA0: case 0xA1: case 0xA2: case 0xA3: { // MOV with a 64-bit absolute address
        size = (op & 1) ? osz(d) : 1;
        uint64_t addr = d->addrsize ? fetch32(d) : fetch64(d);
        FIN();
        addr += seg_base(s, d);
        if (op < 0xA2) set_reg(s, d, X64_RAX, size, mem_read(addr, size));
        else mem_write(addr, size, s->gpr[X64_RAX]);
        return CONT;
    }
    case 0xA4: case 0xA5: case 0xA6: case 0xA7: case 0xAA: case 0xAB:
    case 0xAC: case 0xAD: case 0xAE: case 0xAF:
        FIN();
        return string_op(s, d, op);
    case 0xA8: case 0xA9: // TEST rAX, imm
        size = op == 0xA8 ? 1 : osz(d);
        v = size == 1 ? fetch8(d) : size == 2 ? fetch16(d) : (uint64_t)(int64_t)(int32_t)fetch32(d);
        FIN();
        alu(s, ALU_AND, s->gpr[X64_RAX], v, 0, size);
        return CONT;
    case 0xB0: case 0xB1: case 0xB2: case 0xB3: case 0xB4: case 0xB5: case 0xB6: case 0xB7:
        v = fetch8(d);
        FIN();
        set_reg(s, d, (op & 7) | ((d->rex & 1) << 3), 1, v);
        return CONT;
    case 0xB8: case 0xB9: case 0xBA: case 0xBB: case 0xBC: case 0xBD: case 0xBE: case 0xBF:
        size = osz(d);
        v = size == 8 ? fetch64(d) : size == 2 ? fetch16(d) : fetch32(d);
        FIN();
        set_reg(s, d, (op & 7) | ((d->rex & 1) << 3), size, v);
        return CONT;
    case 0xC0: case 0xC1: case 0xD0: case 0xD1: case 0xD2: case 0xD3: { // shift group
        size = (op & 1) ? osz(d) : 1;
        modrm(d);
        unsigned count = op <= 0xC1 ? fetch8(d) : op <= 0xD1 ? 1 : (unsigned)(s->gpr[X64_RCX] & 0xff);
        FIN();
        unsigned masked = count & (size == 8 ? 63 : 31);
        v = shift(s, d->regf, get_rm(s, d, size), count, size);
        if (masked || size == 4) set_rm(s, d, size, v);
        return CONT;
    }
    case 0xC2: case 0xC3: { // RET
        uint16_t release = op == 0xC2 ? fetch16(d) : 0;
        FIN();
        s->rip = pop(s, 8);
        s->gpr[X64_RSP] += release;
        return CONT;
    }
    case 0xC4: case 0xC5: case 0x62: // VEX, EVEX
        return X64_EXIT_UNSUPPORTED;
    case 0xC6: case 0xC7: // MOV Ev, imm
        size = op == 0xC6 ? 1 : osz(d);
        modrm(d);
        if (d->regf != 0) return X64_EXIT_UNSUPPORTED;   // XABORT, XBEGIN
        v = size == 1 ? fetch8(d) : size == 2 ? fetch16(d) : (uint64_t)(int64_t)(int32_t)fetch32(d);
        FIN();
        set_rm(s, d, size, v);
        return CONT;
    case 0xC8: { // ENTER
        uint16_t frame = fetch16(d);
        uint8_t level = fetch8(d);
        FIN();
        if (level & 31) return X64_EXIT_UNSUPPORTED;
        push(s, s->gpr[X64_RBP], 8);
        s->gpr[X64_RBP] = s->gpr[X64_RSP];
        s->gpr[X64_RSP] -= frame;
        return CONT;
    }
    case 0xC9: // LEAVE
        FIN();
        s->gpr[X64_RSP] = s->gpr[X64_RBP];
        s->gpr[X64_RBP] = pop(s, 8);
        return CONT;
    case 0xCC: // INT3
        FIN();
        return X64_EXIT_BREAKPOINT;
    case 0xCD: // INT n
        d->disp = fetch8(d);   // kept for the exit information
        FIN();
        return X64_EXIT_INTERRUPT;
    case 0xF1: // INT1
        FIN();
        return X64_EXIT_BREAKPOINT;
    case 0xD7: // XLAT
        FIN();
        v = (d->addrsize ? s->gpr[X64_RBX] & 0xffffffffull : s->gpr[X64_RBX]) + (s->gpr[X64_RAX] & 0xff);
        set_reg(s, d, X64_RAX, 1, mem_read(v + seg_base(s, d), 1));
        return CONT;
    case 0xD9: case 0xDB: case 0xDD: case 0xDF: case 0xD8: case 0xDA: case 0xDC: case 0xDE: {
        // x87: only the control and status word instructions C runtimes use.
        modrm(d);
        FIN();
        if (op == 0xD9 && d->mem && d->regf == 5) { s->fcw = (uint16_t)mem_read(d->ea, 2); return CONT; }   // FLDCW
        if (op == 0xD9 && d->mem && d->regf == 7) { mem_write(d->ea, 2, s->fcw); return CONT; }              // FNSTCW
        if (op == 0xDD && d->mem && d->regf == 7) { mem_write(d->ea, 2, s->fsw); return CONT; }              // FNSTSW m16
        if (op == 0xDF && !d->mem && d->regf == 4 && d->rmf == 0) { set_reg(s, d, X64_RAX, 2, s->fsw); return CONT; } // FNSTSW AX
        if (op == 0xDB && !d->mem && d->regf == 4 && d->rmf == 3) { s->fcw = 0x37f; s->fsw = 0; return CONT; }        // FNINIT
        if (op == 0xDB && !d->mem && d->regf == 4 && d->rmf == 2) { s->fsw &= 0x7f00; return CONT; }                  // FNCLEX
        return X64_EXIT_UNSUPPORTED;
    }
    case 0xE0: case 0xE1: case 0xE2: case 0xE3: { // LOOPNE LOOPE LOOP JRCXZ
        t = (uint64_t)(int64_t)(int8_t)fetch8(d);
        FIN();
        int asize = d->addrsize ? 4 : 8;
        uint64_t count = s->gpr[X64_RCX] & mask_of(asize);
        bool take;
        if (op == 0xE3) take = count == 0;
        else {
            count = (count - 1) & mask_of(asize);
            set_reg(s, d, X64_RCX, asize, count);
            take = count != 0;
            if (op == 0xE0) take = take && !(s->rflags & FL_ZF);
            if (op == 0xE1) take = take && (s->rflags & FL_ZF);
        }
        if (take) s->rip = d->next + t;
        return CONT;
    }
    case 0xE8: // CALL rel32
        t = (uint64_t)(int64_t)(int32_t)fetch32(d);
        FIN();
        return call(s, env, d->next + t, d->next);
    case 0xE9: case 0xEB: // JMP
        t = op == 0xE9 ? (uint64_t)(int64_t)(int32_t)fetch32(d) : (uint64_t)(int64_t)(int8_t)fetch8(d);
        FIN();
        return jump(s, env, d->next + t);
    case 0xF4: case 0xFA: case 0xFB: case 0xE4: case 0xE5: case 0xE6: case 0xE7:
    case 0xEC: case 0xED: case 0xEE: case 0xEF: case 0x6C: case 0x6D: case 0x6E: case 0x6F:
        FIN();   // HLT, CLI, STI, IN, OUT, INS, OUTS
        return X64_EXIT_PRIVILEGED;
    case 0xF5: FIN(); s->rflags ^= FL_CF; return CONT;              // CMC
    case 0xF8: FIN(); s->rflags &= ~FL_CF; return CONT;             // CLC
    case 0xF9: FIN(); s->rflags |= FL_CF; return CONT;              // STC
    case 0xFC: FIN(); s->rflags &= ~FL_DF; return CONT;             // CLD
    case 0xFD: FIN(); s->rflags |= FL_DF; return CONT;              // STD
    case 0xF6: case 0xF7: { // group 3
        size = op == 0xF6 ? 1 : osz(d);
        modrm(d);
        if (d->regf < 2) v = size == 1 ? fetch8(d) : size == 2 ? fetch16(d) : (uint64_t)(int64_t)(int32_t)fetch32(d);
        FIN();
        if (d->lock && (d->regf != 2 && d->regf != 3)) return X64_EXIT_ILLEGAL;
        LOCK_MEM();
        switch (d->regf) {
        case 0: case 1: alu(s, ALU_AND, get_rm(s, d, size), v, 0, size); return CONT;   // TEST
        case 2: rmw(s, d, size, RMW_NOT, 0, 0); return CONT;
        case 3: rmw(s, d, size, RMW_NEG, 0, 0); return CONT;
        }
        uint64_t src = get_rm(s, d, size), m = mask_of(size);
        if (d->regf == 4 || d->regf == 5) { // MUL, IMUL
            uint64_t lo, hi;
            bool over;
            if (d->regf == 4) {
                unsigned __int128 p = (unsigned __int128)(s->gpr[X64_RAX] & m) * src;
                lo = (uint64_t)p & m;
                hi = size == 8 ? (uint64_t)(p >> 64) : (uint64_t)(p >> (size * 8)) & m;
                over = hi != 0;
            } else {
                __int128 p = (__int128)sext(s->gpr[X64_RAX], size) * sext(src, size);
                lo = (uint64_t)p & m;
                hi = size == 8 ? (uint64_t)((unsigned __int128)p >> 64) : (uint64_t)(p >> (size * 8)) & m;
                over = (__int128)sext(lo, size) != p;
            }
            if (size == 1) set_reg(s, d, X64_RAX, 2, lo | (hi << 8));
            else { set_reg(s, d, X64_RAX, size, lo); set_reg(s, d, X64_RDX, size, hi); }
            set_flags(s, FL_ARITH, (over ? FL_CF | FL_OF : 0) | szp(lo, size));
            return CONT;
        }
        // DIV, IDIV
        if ((src & m) == 0) return X64_EXIT_DIVIDE_ERROR;
        uint64_t lo = size == 1 ? s->gpr[X64_RAX] & 0xff : s->gpr[X64_RAX] & m;
        uint64_t hi = size == 1 ? (s->gpr[X64_RAX] >> 8) & 0xff : s->gpr[X64_RDX] & m;
        uint64_t q, r;
        if (d->regf == 6) {
            unsigned __int128 n = ((unsigned __int128)hi << (size * 8)) | lo;
            unsigned __int128 qq = n / src;
            if (qq > m) return X64_EXIT_DIVIDE_ERROR;
            q = (uint64_t)qq;
            r = (uint64_t)(n % src);
        } else {
            __int128 n = (__int128)(((unsigned __int128)hi << (size * 8)) | lo);
            int bits = size * 16;
            if (bits < 128) n = (__int128)((unsigned __int128)n << (128 - bits)) >> (128 - bits);  // sign-extend
            int64_t dv = sext(src, size);
            __int128 qq;
            if (dv == -1) {
                // n / -1 overflows __int128 itself for its minimum, which only size 8 reaches.
                if ((unsigned __int128)n == (unsigned __int128)1 << 127) return X64_EXIT_DIVIDE_ERROR;
                qq = -n;
                r = 0;
            } else {
                qq = n / dv;
                r = (uint64_t)(n % dv) & m;
            }
            if (qq > (__int128)(int64_t)(sign_of(size) - 1) || qq < -(__int128)sign_of(size)) return X64_EXIT_DIVIDE_ERROR;
            q = (uint64_t)qq & m;
        }
        if (size == 1) set_reg(s, d, X64_RAX, 2, q | (r << 8));
        else { set_reg(s, d, X64_RAX, size, q); set_reg(s, d, X64_RDX, size, r); }
        return CONT;
    }
    case 0xFE: case 0xFF: { // group 4, group 5
        modrm(d);
        FIN();
        size = op == 0xFE ? 1 : osz(d);
        if (d->lock && d->regf > 1) return X64_EXIT_ILLEGAL;
        switch (d->regf) {
        case 0: LOCK_MEM(); rmw(s, d, size, RMW_INC, 0, 0); return CONT;
        case 1: LOCK_MEM(); rmw(s, d, size, RMW_DEC, 0, 0); return CONT;
        }
        if (op == 0xFE) return X64_EXIT_ILLEGAL;
        switch (d->regf) {
        case 2: return call(s, env, get_rm(s, d, 8), d->next);
        case 4: return jump(s, env, get_rm(s, d, 8));
        case 6:
            size = d->opsize ? 2 : 8;
            push(s, get_rm(s, d, size), size);
            return CONT;
        default: return X64_EXIT_UNSUPPORTED;   // far CALL and JMP
        }
    }
    default:
        return X64_EXIT_ILLEGAL;   // the opcodes 64-bit mode removed (06, 07, 27, 60, 9A, D4...)
    }

    if (map != 1) return sse(s, d, map, op);

    switch (op) {
    case 0x05: FIN(); return X64_EXIT_SYSCALL;
    case 0x0B: FIN(); return X64_EXIT_ILLEGAL;   // UD2
    case 0x0D: case 0x18: case 0x19: case 0x1A: case 0x1B: case 0x1C: case 0x1D: case 0x1E: case 0x1F:
        modrm(d);   // prefetches and the hint NOPs (ENDBR64 is F3 0F 1E FA)
        FIN();
        return CONT;
    case 0x31: // RDTSC
        FIN();
        t = rdtsc();
        s->gpr[X64_RAX] = t & 0xffffffffull;
        s->gpr[X64_RDX] = t >> 32;
        return CONT;
    case 0x40: case 0x41: case 0x42: case 0x43: case 0x44: case 0x45: case 0x46: case 0x47:
    case 0x48: case 0x49: case 0x4A: case 0x4B: case 0x4C: case 0x4D: case 0x4E: case 0x4F: // CMOVcc
        size = osz(d);
        modrm(d);
        FIN();
        v = get_rm(s, d, size);
        set_reg(s, d, d->reg, size, cond(s, op & 15) ? v : get_reg(s, d, d->reg, size));
        return CONT;
    case 0x80: case 0x81: case 0x82: case 0x83: case 0x84: case 0x85: case 0x86: case 0x87:
    case 0x88: case 0x89: case 0x8A: case 0x8B: case 0x8C: case 0x8D: case 0x8E: case 0x8F: // Jcc rel32
        t = (uint64_t)(int64_t)(int32_t)fetch32(d);
        FIN();
        if (cond(s, op & 15)) s->rip = d->next + t;
        return CONT;
    case 0x90: case 0x91: case 0x92: case 0x93: case 0x94: case 0x95: case 0x96: case 0x97:
    case 0x98: case 0x99: case 0x9A: case 0x9B: case 0x9C: case 0x9D: case 0x9E: case 0x9F: // SETcc
        modrm(d);
        FIN();
        set_rm(s, d, 1, cond(s, op & 15));
        return CONT;
    case 0xA2: { // CPUID
        FIN();
        uint32_t out[4];
        x64_cpuid((uint32_t)s->gpr[X64_RAX], (uint32_t)s->gpr[X64_RCX], out);
        s->gpr[X64_RAX] = out[0]; s->gpr[X64_RBX] = out[1];
        s->gpr[X64_RCX] = out[2]; s->gpr[X64_RDX] = out[3];
        return CONT;
    }
    case 0xA3: case 0xAB: case 0xB3: case 0xBB: case 0xBA: { // BT BTS BTR BTC
        size = osz(d);
        modrm(d);
        int kind;
        if (op == 0xBA) {
            if (d->regf < 4) return X64_EXIT_ILLEGAL;
            v = fetch8(d) & (uint64_t)(size * 8 - 1);
            kind = d->regf - 4;
        } else {
            kind = op == 0xA3 ? 0 : op == 0xAB ? 1 : op == 0xB3 ? 2 : 3;
        }
        FIN();
        if (kind == 0 && d->lock) return X64_EXIT_ILLEGAL;
        LOCK_MEM();
        if (op != 0xBA) {
            int64_t offset = sext(get_reg(s, d, d->reg, size), size);
            if (d->mem) {
                // A register bit offset reaches outside the operand.
                d->ea += (uint64_t)((offset >> (size == 8 ? 6 : size == 4 ? 5 : 4)) * size);
                offset &= size * 8 - 1;
            }
            v = (uint64_t)offset & (uint64_t)(size * 8 - 1);
        }
        if (kind == 0) {
            set_flags(s, FL_CF, (get_rm(s, d, size) >> v) & 1);
            return CONT;
        }
        rmw(s, d, size, kind == 1 ? RMW_BTS : kind == 2 ? RMW_BTR : RMW_BTC, 0, v);
        return CONT;
    }
    case 0xA4: case 0xA5: case 0xAC: case 0xAD: { // SHLD SHRD
        size = osz(d);
        modrm(d);
        unsigned count = (op & 1) ? (unsigned)(s->gpr[X64_RCX] & 0xff) : fetch8(d);
        FIN();
        unsigned masked = count & (size == 8 ? 63 : 31);
        v = double_shift(s, op < 0xA8, get_rm(s, d, size), get_reg(s, d, d->reg, size), count, size);
        if (masked || size == 4) set_rm(s, d, size, v);
        return CONT;
    }
    case 0xAE: // group 15
        modrm(d);
        FIN();
        if (d->mem) {
            if (d->rep || (d->opsize && d->regf != 7)) return X64_EXIT_UNSUPPORTED;
            switch (d->regf) {
            case 0: fxsave(s, d->ea); return CONT;
            case 1: fxrstor(s, d->ea); return CONT;
            case 2: set_mxcsr(s, (uint32_t)mem_read(d->ea, 4)); return CONT;   // LDMXCSR
            case 3: mem_write(d->ea, 4, s->mxcsr); return CONT;                // STMXCSR
            case 7: return CONT;                                               // CLFLUSH, CLFLUSHOPT
            default: return X64_EXIT_UNSUPPORTED;                              // XSAVE family
            }
        }
        if (!d->rep && !d->opsize && d->regf >= 5) {   // LFENCE MFENCE SFENCE
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
            return CONT;
        }
        return X64_EXIT_UNSUPPORTED;   // RDFSBASE and the rest
    case 0xAF: { // IMUL Gv, Ev
        size = osz(d);
        modrm(d);
        FIN();
        __int128 full = (__int128)sext(get_reg(s, d, d->reg, size), size) * sext(get_rm(s, d, size), size);
        uint64_t r = (uint64_t)full & mask_of(size);
        bool over = (__int128)sext(r, size) != full;
        set_flags(s, FL_ARITH, (over ? FL_CF | FL_OF : 0) | szp(r, size));
        set_reg(s, d, d->reg, size, r);
        return CONT;
    }
    case 0xB0: case 0xB1: { // CMPXCHG
        size = op == 0xB0 ? 1 : osz(d);
        modrm(d);
        FIN();
        LOCK_MEM();
        uint64_t acc = get_reg(s, d, X64_RAX, size), src = get_reg(s, d, d->reg, size);
        if (d->mem && d->lock) {
            uint64_t old = mem_read(d->ea, size);
            for (;;) {
                op_sub(s, acc, old, 0, size);
                if (old != acc) { set_reg(s, d, X64_RAX, size, old); break; }
                if (mem_cas(d->ea, size, &old, src)) break;
            }
            return CONT;
        }
        uint64_t old = get_rm(s, d, size);
        op_sub(s, acc, old, 0, size);
        if (old == acc) set_rm(s, d, size, src);
        else set_reg(s, d, X64_RAX, size, old);   // a register destination keeps its upper half
        return CONT;
    }
    case 0xB6: case 0xB7: case 0xBE: case 0xBF: { // MOVZX MOVSX
        size = osz(d);
        modrm(d);
        FIN();
        int from = (op & 1) ? 2 : 1;
        v = get_rm(s, d, from);
        set_reg(s, d, d->reg, size, op >= 0xBE ? (uint64_t)sext(v, from) : v);
        return CONT;
    }
    case 0xB8: // POPCNT
        if (d->rep != 0xF3) return X64_EXIT_UNSUPPORTED;   // JMPE
        size = osz(d);
        modrm(d);
        FIN();
        v = get_rm(s, d, size);
        set_flags(s, FL_ARITH, v ? 0 : FL_ZF);
        set_reg(s, d, d->reg, size, (uint64_t)__builtin_popcountll(v));
        return CONT;
    case 0xBC: case 0xBD: { // BSF BSR TZCNT LZCNT
        size = osz(d);
        modrm(d);
        FIN();
        v = get_rm(s, d, size);
        int bits = size * 8;
        if (d->rep == 0xF3) {
            uint64_t r = v == 0 ? (uint64_t)bits
                       : op == 0xBC ? (uint64_t)__builtin_ctzll(v) : (uint64_t)(__builtin_clzll(v) - (64 - bits));
            set_flags(s, FL_CF | FL_ZF, (v == 0 ? FL_CF : 0) | (r == 0 ? FL_ZF : 0));
            set_reg(s, d, d->reg, size, r);
            return CONT;
        }
        if (v == 0) { set_flags(s, FL_ZF, FL_ZF); return CONT; }   // destination unchanged
        set_flags(s, FL_ZF, 0);
        set_reg(s, d, d->reg, size, op == 0xBC ? (uint64_t)__builtin_ctzll(v) : (uint64_t)(63 - __builtin_clzll(v)));
        return CONT;
    }
    case 0xC0: case 0xC1: { // XADD
        size = op == 0xC0 ? 1 : osz(d);
        modrm(d);
        FIN();
        LOCK_MEM();
        uint64_t src = get_reg(s, d, d->reg, size);
        if (d->mem && d->lock) {
            uint64_t old = mem_read(d->ea, size);
            uint64_t sum;
            for (;;) {
                sum = op_add(s, old, src, 0, size);
                if (mem_cas(d->ea, size, &old, sum)) break;
            }
            set_reg(s, d, d->reg, size, old);
            return CONT;
        }
        uint64_t old = get_rm(s, d, size);
        uint64_t sum = op_add(s, old, src, 0, size);
        set_reg(s, d, d->reg, size, old);
        set_rm(s, d, size, sum);
        return CONT;
    }
    case 0xC3: // MOVNTI
        size = osz(d);
        modrm(d);
        FIN();
        NEED_MEM();
        mem_write(d->ea, size, get_reg(s, d, d->reg, size));
        return CONT;
    case 0xC7: { // CMPXCHG8B CMPXCHG16B
        modrm(d);
        FIN();
        if (d->regf != 1 || !d->mem) return X64_EXIT_UNSUPPORTED;   // RDRAND, RDSEED, ...
        if (d->rex & 8) {
            if (d->ea & 15) return X64_EXIT_ILLEGAL;   // #GP on x86
            unsigned __int128 expected = ((unsigned __int128)s->gpr[X64_RDX] << 64) | s->gpr[X64_RAX];
            unsigned __int128 desired = ((unsigned __int128)s->gpr[X64_RCX] << 64) | s->gpr[X64_RBX];
            bool ok = __atomic_compare_exchange_n((unsigned __int128 *)(uintptr_t)d->ea, &expected, desired,
                                                  false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
            if (!ok) { s->gpr[X64_RAX] = (uint64_t)expected; s->gpr[X64_RDX] = (uint64_t)(expected >> 64); }
            set_flags(s, FL_ZF, ok ? FL_ZF : 0);
        } else {
            uint64_t expected = (s->gpr[X64_RDX] << 32) | (s->gpr[X64_RAX] & 0xffffffffull);
            uint64_t desired = (s->gpr[X64_RCX] << 32) | (s->gpr[X64_RBX] & 0xffffffffull);
            bool ok = mem_cas(d->ea, 8, &expected, desired);
            if (!ok) { s->gpr[X64_RAX] = expected & 0xffffffffull; s->gpr[X64_RDX] = expected >> 32; }
            set_flags(s, FL_ZF, ok ? FL_ZF : 0);
        }
        return CONT;
    }
    case 0xC8: case 0xC9: case 0xCA: case 0xCB: case 0xCC: case 0xCD: case 0xCE: case 0xCF: { // BSWAP
        FIN();
        int r = (op & 7) | ((d->rex & 1) << 3);
        if (d->rex & 8) s->gpr[r] = __builtin_bswap64(s->gpr[r]);
        else if (d->opsize) set_reg(s, d, r, 2, 0);   // undefined on x86; zero, as Intel's hardware gives
        else s->gpr[r] = __builtin_bswap32((uint32_t)s->gpr[r]);
        return CONT;
    }
    case 0x10: case 0x11: case 0x12: case 0x13: case 0x14: case 0x15: case 0x16: case 0x17:
    case 0x28: case 0x29: case 0x2A: case 0x2B: case 0x2C: case 0x2D: case 0x2E: case 0x2F:
    case 0x50: case 0x51: case 0x52: case 0x53: case 0x54: case 0x55: case 0x56: case 0x57:
    case 0x58: case 0x59: case 0x5A: case 0x5B: case 0x5C: case 0x5D: case 0x5E: case 0x5F:
    case 0xC2: case 0xC4: case 0xC5: case 0xC6:
        return sse(s, d, map, op);
    default:
        if ((op >= 0x60 && op <= 0x7F) || op >= 0xD0) return sse(s, d, map, op);
        return X64_EXIT_UNSUPPORTED;   // system instructions, PUSH/POP FS/GS, ...
    }
}

// ---------------------------------------------------------------------------
// Public interface

void x64_state_init(X64State *s)
{
    memset(s, 0, sizeof *s);
    s->rflags = 0x202;
    s->mxcsr = 0x1f80;
    s->fcw = 0x27f;
    s->seg_cs = 0x33;
    s->seg_ds = s->seg_es = s->seg_ss = s->seg_gs = 0x2b;
    s->seg_fs = 0x53;
}

X64Exit x64_run(X64State *s, const X64Env *env, uint64_t budget, X64ExitInfo *info)
{
    int saved_rounding = fegetround();
    fesetround(host_rounding(s->mxcsr));
    Dec d;
    int r;
    for (;;) {
        if (s->rip == env->stop_rip || budget == 0) {
            r = s->rip == env->stop_rip ? X64_EXIT_STOP : X64_EXIT_BUDGET;
            if (info) { memset(info, 0, sizeof *info); info->rip = s->rip; }
            break;
        }
        if (budget != UINT64_MAX) budget--;
        uint64_t start = s->rip;
        r = step(s, env, &d);
        if (r == CONT) { s->instructions++; continue; }
        if (r == X64_EXIT_DIVIDE_ERROR || r == X64_EXIT_ILLEGAL || r == X64_EXIT_PRIVILEGED ||
            r == X64_EXIT_UNSUPPORTED)
            s->rip = start;
        else
            s->instructions++;
        if (info) {
            size_t length = (size_t)(d.p - d.begin);
            if (length > sizeof info->bytes) length = sizeof info->bytes;
            memset(info, 0, sizeof *info);
            info->rip = start;
            info->length = (uint8_t)length;
            memcpy(info->bytes, (const void *)(uintptr_t)start, length);
            if (r == X64_EXIT_INTERRUPT) info->vector = (uint8_t)d.disp;
        }
        break;
    }
    fesetround(saved_rounding);
    return (X64Exit)r;
}

void x64_return_from_native(X64State *s, uint64_t rax)
{
    s->gpr[X64_RAX] = rax;
    s->rip = pop(s, 8);
}

void x64_cpuid(uint32_t leaf, uint32_t subleaf, uint32_t out[4])
{
    static const char brand[48] = "Madeira x86-64 interpreter";
    (void)subleaf;
    out[0] = out[1] = out[2] = out[3] = 0;
    switch (leaf) {
    case 0:
        out[0] = 7;
        memcpy(&out[1], "Genu", 4);   // EBX EDX ECX: GenuineIntel
        memcpy(&out[3], "ineI", 4);
        memcpy(&out[2], "ntel", 4);
        break;
    case 1:
        out[0] = 0x000006f0;                            // family 6
        out[1] = (1u << 16) | (8u << 8);                // one logical processor, 64-byte CLFLUSH line
        out[2] = (1u << 0) | (1u << 13) | (1u << 23);   // SSE3, CMPXCHG16B, POPCNT
        out[3] = (1u << 0) | (1u << 4) | (1u << 8) | (1u << 15) | (1u << 19)
               | (1u << 24) | (1u << 25) | (1u << 26);  // FPU TSC CX8 CMOV CLFSH FXSR SSE SSE2
        break;
    case 0x80000000:
        out[0] = 0x80000004;
        break;
    case 0x80000001:
        out[2] = (1u << 0) | (1u << 5);                 // LAHF/SAHF, LZCNT
        out[3] = (1u << 11) | (1u << 20) | (1u << 29);  // SYSCALL, NX, long mode
        break;
    case 0x80000002: case 0x80000003: case 0x80000004:
        memcpy(out, brand + 16 * (leaf - 0x80000002), 16);
        break;
    }
}

const char *x64_exit_name(X64Exit e)
{
    switch (e) {
    case X64_EXIT_STOP: return "stop";
    case X64_EXIT_BUDGET: return "budget";
    case X64_EXIT_NATIVE: return "native";
    case X64_EXIT_SYSCALL: return "syscall";
    case X64_EXIT_BREAKPOINT: return "breakpoint";
    case X64_EXIT_INTERRUPT: return "interrupt";
    case X64_EXIT_DIVIDE_ERROR: return "divide-error";
    case X64_EXIT_ILLEGAL: return "illegal";
    case X64_EXIT_PRIVILEGED: return "privileged";
    case X64_EXIT_UNSUPPORTED: return "unsupported";
    }
    return "?";
}

// AMD64 CONTEXT offsets (winnt.h).
enum {
    CTX_MXCSR = 0x34, CTX_SEGCS = 0x38, CTX_SEGDS = 0x3a, CTX_SEGES = 0x3c, CTX_SEGFS = 0x3e,
    CTX_SEGGS = 0x40, CTX_SEGSS = 0x42, CTX_EFLAGS = 0x44, CTX_RAX = 0x78, CTX_RIP = 0xf8,
    CTX_FLTSAVE = 0x100, CTX_FLT_MXCSR = CTX_FLTSAVE + 24, CTX_XMM0 = CTX_FLTSAVE + 160,
};

void x64_state_from_context(X64State *s, const void *context)
{
    const uint8_t *c = (const uint8_t *)context;
    uint32_t flags32;
    memcpy(s->gpr, c + CTX_RAX, sizeof s->gpr);
    memcpy(&s->rip, c + CTX_RIP, 8);
    memcpy(&flags32, c + CTX_EFLAGS, 4);
    s->rflags = flags32 | 2;
    memcpy(&s->mxcsr, c + CTX_MXCSR, 4);
    memcpy(&s->seg_cs, c + CTX_SEGCS, 2);
    memcpy(&s->seg_ds, c + CTX_SEGDS, 2);
    memcpy(&s->seg_es, c + CTX_SEGES, 2);
    memcpy(&s->seg_fs, c + CTX_SEGFS, 2);
    memcpy(&s->seg_gs, c + CTX_SEGGS, 2);
    memcpy(&s->seg_ss, c + CTX_SEGSS, 2);
    memcpy(&s->fcw, c + CTX_FLTSAVE + 0, 2);
    memcpy(&s->fsw, c + CTX_FLTSAVE + 2, 2);
    memcpy(s->xmm, c + CTX_XMM0, sizeof s->xmm);
}

void x64_state_to_context(const X64State *s, void *context)
{
    uint8_t *c = (uint8_t *)context;
    uint32_t flags32 = (uint32_t)s->rflags;
    memcpy(c + CTX_RAX, s->gpr, sizeof s->gpr);
    memcpy(c + CTX_RIP, &s->rip, 8);
    memcpy(c + CTX_EFLAGS, &flags32, 4);
    memcpy(c + CTX_MXCSR, &s->mxcsr, 4);
    memcpy(c + CTX_FLT_MXCSR, &s->mxcsr, 4);
    memcpy(c + CTX_SEGCS, &s->seg_cs, 2);
    memcpy(c + CTX_SEGDS, &s->seg_ds, 2);
    memcpy(c + CTX_SEGES, &s->seg_es, 2);
    memcpy(c + CTX_SEGFS, &s->seg_fs, 2);
    memcpy(c + CTX_SEGGS, &s->seg_gs, 2);
    memcpy(c + CTX_SEGSS, &s->seg_ss, 2);
    memcpy(c + CTX_FLTSAVE + 0, &s->fcw, 2);
    memcpy(c + CTX_FLTSAVE + 2, &s->fsw, 2);
    memcpy(c + CTX_XMM0, s->xmm, sizeof s->xmm);
}
