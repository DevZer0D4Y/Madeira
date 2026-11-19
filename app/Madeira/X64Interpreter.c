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

#include <math.h>
#include <string.h>
#include <time.h>
#if !defined(__x86_64__) && !defined(__i386__) && !defined(__aarch64__)
#include <fenv.h>
#endif

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
    uint64_t start, next;       // guest addresses
    const uint8_t *begin, *p;   // host pointers to the bytes
    bool m32;                   // 32-bit mode
    uint64_t gbase;             // guest_base, in 32-bit mode
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
    bool addr16;            // a 16-bit memory operand (32-bit mode with 67): unsupported
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
    d->addr16 = d->m32 && d->addrsize;
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
        d->riprel = !d->m32;   // 32-bit mode: an absolute disp32
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
    if (length > 15 || d->addr16) return false;
    d->next = d->start + length;
    if (d->m32) d->next = (uint32_t)d->next;
    if (d->mem) {
        uint64_t off = (uint64_t)d->disp;
        if (d->riprel) off += d->next;
        if (d->base >= 0) off += s->gpr[d->base];
        if (d->index >= 0) off += s->gpr[d->index] << d->scale;
        if (d->addrsize || d->m32) off &= 0xffffffffull;
        d->off = off;
        uint64_t a = off + (d->seg == 4 ? s->fs_base : d->seg == 5 ? s->gs_base : 0);
        d->ea = d->m32 ? d->gbase + (uint32_t)a : a;
    }
    return true;
}

static inline int osz(const Dec *d) { return (d->rex & 8) ? 8 : d->opsize ? 2 : 4; }

// ---------------------------------------------------------------------------
// Registers and memory

// By operand size (1, 2, 4 or 8 bytes).
static const uint64_t size_masks[16] = { [1] = 0xff, [2] = 0xffff, [4] = 0xffffffffull, [8] = ~0ull };
static const uint64_t size_signs[16] = { [1] = 0x80, [2] = 0x8000, [4] = 0x80000000ull, [8] = 0x8000000000000000ull };
static inline uint64_t mask_of(int size) { return size_masks[size & 15]; }

// Guest address to host address, and the width of addresses and of EIP.
static inline uint64_t ha(const X64State *s, uint64_t guest)
{
    return s->mode32 ? s->guest_base + (uint32_t)guest : guest;
}
static inline uint64_t ipm(const X64State *s, uint64_t v) { return s->mode32 ? (uint32_t)v : v; }
// PUSH and POP's operand size: 64 bits (32 in 32-bit mode), 16 with 66.
static inline int stack_size(const X64State *s, const Dec *d) { return d->opsize ? 2 : s->mode32 ? 4 : 8; }
static inline uint64_t sign_of(int size) { return size_signs[size & 15]; }

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

// The store comes first, so a fault leaves RSP as it was.
static inline void push(X64State *s, uint64_t v, int size)
{
    uint64_t sp = ipm(s, s->gpr[X64_RSP] - (uint64_t)size);
    mem_write(ha(s, sp), size, v);
    s->gpr[X64_RSP] = sp;
}

static inline uint64_t pop(X64State *s, int size)
{
    uint64_t v = mem_read(ha(s, s->gpr[X64_RSP]), size);
    s->gpr[X64_RSP] = ipm(s, s->gpr[X64_RSP] + (uint64_t)size);
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

// (hi:lo) / d with hi < d, so the quotient fits 64 bits: long division a bit at
// a time, which needs no compiler runtime (a 128-bit "/" does on ARM64 Windows).
static uint64_t udiv128(uint64_t hi, uint64_t lo, uint64_t d, uint64_t *remainder)
{
    uint64_t q = 0;
    for (int i = 0; i < 64; i++) {
        bool carry = (hi >> 63) != 0;
        hi = (hi << 1) | (lo >> 63);
        lo <<= 1;
        q <<= 1;
        if (carry || hi >= d) { hi -= d; q |= 1; }
    }
    *remainder = hi;
    return q;
}

// A whole register of 16, 32 or 64 bits (rAX, rDX: no byte registers).
static inline void put_reg(X64State *s, int reg, int size, uint64_t v)
{
    if (size == 2) s->gpr[reg] = (s->gpr[reg] & ~0xffffull) | (v & 0xffff);
    else if (size == 4) s->gpr[reg] = v & 0xffffffffull;
    else s->gpr[reg] = v;
}

// MUL, IMUL, DIV and IDIV (group 3, regf 4 to 7) of rAX (and rDX) by src: CONT or
// X64_EXIT_DIVIDE_ERROR, which changes nothing. DIV and IDIV leave the flags.
static int muldiv(X64State *s, int regf, uint64_t src, int size)
{
    uint64_t m = mask_of(size);
    if (regf == 4 || regf == 5) { // MUL, IMUL
        uint64_t lo, hi;
        bool over;
        if (regf == 4) {
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
        if (size == 1) put_reg(s, X64_RAX, 2, lo | (hi << 8));
        else { put_reg(s, X64_RAX, size, lo); put_reg(s, X64_RDX, size, hi); }
        s->lazy_op = 0;   // every arithmetic flag is set here
        set_flags(s, FL_ARITH, (over ? FL_CF | FL_OF : 0) | szp(lo, size));
        return CONT;
    }
    // DIV, IDIV
    if ((src & m) == 0) return X64_EXIT_DIVIDE_ERROR;
    uint64_t lo = size == 1 ? s->gpr[X64_RAX] & 0xff : s->gpr[X64_RAX] & m;
    uint64_t hi = size == 1 ? (s->gpr[X64_RAX] >> 8) & 0xff : s->gpr[X64_RDX] & m;
    uint64_t q, r;
    int bits = size * 8;
    if (size < 8) {
        // The dividend fits 64 bits.
        if (regf == 6) {
            uint64_t n = (hi << bits) | lo;
            q = n / src;
            r = n % src;
            if (q > m) return X64_EXIT_DIVIDE_ERROR;
        } else {
            int64_t n = sext((hi << bits) | lo, size * 2), dv = sext(src, size);
            if (dv == -1 && n == INT64_MIN) return X64_EXIT_DIVIDE_ERROR;
            int64_t qq = n / dv;
            if (qq > (int64_t)(sign_of(size) - 1) || qq < -(int64_t)sign_of(size)) return X64_EXIT_DIVIDE_ERROR;
            q = (uint64_t)qq & m;
            r = (uint64_t)(n % dv) & m;
        }
    } else if (regf == 6) {
        if (hi >= src) return X64_EXIT_DIVIDE_ERROR;   // the quotient would not fit
        q = udiv128(hi, lo, src, &r);
    } else {
        // On magnitudes; the remainder takes the dividend's sign.
        bool neg_n = (hi >> 63) != 0, neg_d = (src >> 63) != 0;
        uint64_t mhi = hi, mlo = lo, md = neg_d ? 0 - src : src;
        if (neg_n) { mhi = ~hi + (lo == 0); mlo = 0 - lo; }
        if (mhi >= md) return X64_EXIT_DIVIDE_ERROR;
        uint64_t mq = udiv128(mhi, mlo, md, &r);
        bool neg_q = neg_n != neg_d;
        if (mq > (neg_q ? 0x8000000000000000ull : 0x7fffffffffffffffull)) return X64_EXIT_DIVIDE_ERROR;
        q = neg_q ? 0 - mq : mq;
        if (neg_n) r = 0 - r;
    }
    if (size == 1) put_reg(s, X64_RAX, 2, q | (r << 8));
    else { put_reg(s, X64_RAX, size, q); put_reg(s, X64_RDX, size, r); }
    return CONT;
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

// ---------------------------------------------------------------------------
// The host: its rounding mode, a clock, and the libm functions ntdll lacks.
//
// The interpreter also builds into a Windows DLL that may import nothing but
// ntdll (the WoW64 CPU module, build/wow64-interp). So it sets the rounding mode
// itself, reads time through NtQueryPerformanceCounter there, and has its own
// trunc, nearbyint, fmod and remainder, which are exact and so agree on every
// host. sin, cos, tan, atan2, log, pow and sqrt come from libm, or from ntdll's
// exports of the same functions.

// The host's rounding mode, in x86's RC encoding: 0 nearest, 1 down, 2 up, 3 toward zero.
static int host_get_rc(void)
{
#if defined(__x86_64__) || defined(__i386__)
    uint32_t m;
    __asm__ volatile("stmxcsr %0" : "=m"(m) : : "memory");
    return (int)((m >> 13) & 3);
#elif defined(__aarch64__)
    uint64_t f;   // FPCR.RMode: 0 nearest, 1 up, 2 down, 3 toward zero
    __asm__ volatile("mrs %0, fpcr" : "=r"(f) : : "memory");
    int r = (int)((f >> 22) & 3);
    return r == 1 ? 2 : r == 2 ? 1 : r;
#else
    int r = fegetround();
    return r == FE_DOWNWARD ? 1 : r == FE_UPWARD ? 2 : r == FE_TOWARDZERO ? 3 : 0;
#endif
}

static void host_set_rc(int rc)
{
#if defined(__x86_64__) || defined(__i386__)
    uint32_t m;
    __asm__ volatile("stmxcsr %0" : "=m"(m) : : "memory");
    m = (m & ~0x6000u) | ((uint32_t)(rc & 3) << 13);
    __asm__ volatile("ldmxcsr %0" : : "m"(m) : "memory");
#elif defined(__aarch64__)
    uint64_t f;
    int r = rc == 1 ? 2 : rc == 2 ? 1 : rc & 3;
    __asm__ volatile("mrs %0, fpcr" : "=r"(f) : : "memory");
    f = (f & ~(3ull << 22)) | ((uint64_t)r << 22);
    __asm__ volatile("msr fpcr, %0" : : "r"(f) : "memory");
#else
    static const int modes[4] = { FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO };
    fesetround(modes[rc & 3]);
#endif
}

#if defined(_WIN32)
extern long __stdcall NtQueryPerformanceCounter(long long *counter, long long *frequency);
#endif

static uint64_t host_nanoseconds(void)
{
#if defined(_WIN32)
    long long counter = 0, frequency = 1;
    NtQueryPerformanceCounter(&counter, &frequency);
    return (uint64_t)((double)counter * (1e9 / (double)frequency));
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
}

static double x64m_trunc(double x)
{
    uint64_t u = d2u(x);
    int e = (int)((u >> 52) & 0x7ff) - 1023;
    if (e >= 52) return x;                                // integral, infinite or NaN
    if (e < 0) return u2d(u & 0x8000000000000000ull);     // below 1: a signed zero
    return u2d(u & ~(0x000fffffffffffffull >> e));
}

// The integer the host's rounding mode gives: adding and taking away 2^52 rounds
// any |x| < 2^52 to a whole number in that mode.
static double x64m_nearbyint(double x)
{
    if (!(fabs(x) < 0x1p52)) return x;
    volatile double big = x < 0 ? -0x1p52 : 0x1p52;
    volatile double r = (x + big) - big;
    return r == 0 ? copysign(0.0, x) : r;
}

// Exact, as IEEE 754 defines it: the significands divided bit by bit.
static double x64m_fmod(double x, double y)
{
    uint64_t ux = d2u(x), uy = d2u(y);
    int ex = (int)((ux >> 52) & 0x7ff), ey = (int)((uy >> 52) & 0x7ff);
    uint64_t sign = ux & 0x8000000000000000ull;
    if (!(uy << 1) || isnan(y) || ex == 0x7ff) return (x * y) / (x * y);
    if ((ux << 1) <= (uy << 1)) return (ux << 1) == (uy << 1) ? 0 * x : x;
    if (!ex) { for (uint64_t i = ux << 12; !(i >> 63); i <<= 1) ex--; ux <<= -ex + 1; }
    else ux = (ux & (~0ull >> 12)) | (1ull << 52);
    if (!ey) { for (uint64_t i = uy << 12; !(i >> 63); i <<= 1) ey--; uy <<= -ey + 1; }
    else uy = (uy & (~0ull >> 12)) | (1ull << 52);
    for (; ex > ey; ex--) {
        uint64_t i = ux - uy;
        if (!(i >> 63)) { if (!i) return 0 * x; ux = i; }
        ux <<= 1;
    }
    uint64_t i = ux - uy;
    if (!(i >> 63)) { if (!i) return 0 * x; ux = i; }
    for (; !(ux >> 52); ux <<= 1) ex--;
    if (ex > 0) ux = (ux - (1ull << 52)) | ((uint64_t)ex << 52);
    else ux >>= -ex + 1;
    return u2d(ux | sign);
}

// IEEE remainder: x - n*y with n the nearest integer (even on a tie).
static double x64m_remainder(double x, double y)
{
    if (isnan(x) || isnan(y) || isinf(x) || y == 0) return (x * y) / (x * y);
    bool neg = signbit(x);
    double ay = fabs(y), r = fabs(x);
    if (ay <= 0x1p1022) r = x64m_fmod(r, ay + ay);   // |r| < 2|y|
    if (ay < 0x1p-1021) {
        if (r + r > ay) { r -= ay; if (r + r >= ay) r -= ay; }
    } else {
        double half = 0.5 * ay;
        if (r > half) { r -= ay; if (r >= half) r -= ay; }
    }
    return neg ? -r : r;
}

static const double LN2 = 0.69314718055994530942;

static double x64m_exp2m1(double x)   // 2^x - 1, for F2XM1's |x| <= 1
{
    double t = x * LN2;
    if (fabs(t) < 1e-5) return t + t * t * 0.5 + t * t * t * (1.0 / 6);
    return pow(2.0, x) - 1.0;
}

static double x64m_log1p(double x)
{
    volatile double u = 1.0 + x;
    if (u == 1.0) return x;
    return log(u) * x / (u - 1.0);
}


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
    double r = truncate ? x64m_trunc(v) : x64m_nearbyint(v);
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

static void set_mxcsr(X64State *s, uint32_t v)
{
    s->mxcsr = v & 0xffffu;
    host_set_rc((int)((s->mxcsr >> 13) & 3));
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
            for (int j = 0; j < 8; j++) {
                int diff = a->b[8 * i + j] - b->b[8 * i + j];
                sum += (unsigned)(diff < 0 ? -diff : diff);
            }
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

static int mmx(X64State *s, Dec *d, uint8_t op);
static int sse_body(X64State *s, Dec *d, uint8_t op, int pfx, uint8_t imm);
static int mmx_body(X64State *s, Dec *d, uint8_t op, uint8_t imm);
static inline uint64_t mm_get(const X64State *s, int i) { return s->st[i & 7].mantissa; }
static inline void mm_set(X64State *s, int i, uint64_t v) { s->st[i & 7].mantissa = v; s->st[i & 7].exponent = 0xffff; }
// Any MMX instruction but EMMS: TOP 0 and every register in use.
static inline void mmx_enter(X64State *s) { s->fsw &= (uint16_t)~0x3800; s->ftw = 0xff; }

static int sse(X64State *s, Dec *d, int map, uint8_t op)
{
    // The mandatory prefix: 0 none, 1 66, 2 F3, 3 F2 (F2/F3 win over 66).
    int pfx = d->rep == 0xF3 ? 2 : d->rep == 0xF2 ? 3 : d->opsize ? 1 : 0;
    int i;
    uint8_t imm = 0;

    if (map == 2) {
        if (op != 0x00 || pfx != 1) return X64_EXIT_UNSUPPORTED;
        modrm(d);
        if (!finish(s, d)) return d->addr16 ? X64_EXIT_UNSUPPORTED : X64_EXIT_ILLEGAL;
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
        if (!finish(s, d)) return d->addr16 ? X64_EXIT_UNSUPPORTED : X64_EXIT_ILLEGAL;
        s->rip = d->next;
        X64Vec b = xm(s, d, 16), *a = &s->xmm[d->reg], r = {{0}};   // PALIGNR
        uint8_t cat[32];
        memcpy(cat, b.b, 16);
        memcpy(cat + 16, a->b, 16);
        for (i = 0; i < 16; i++) r.b[i] = (unsigned)(i + imm) < 32 ? cat[i + imm] : 0;
        *a = r;
        return CONT;
    }

    // Opcodes whose 66-less form is MMX.
    bool sse_int = (op >= 0x60 && op <= 0x7F) || op >= 0xD0 || op == 0xC4 || op == 0xC5;
    if (sse_int && pfx == 0) return mmx(s, d, op);

    modrm(d);
    if (op == 0x70 || op == 0x71 || op == 0x72 || op == 0x73 || op == 0xC2 || op == 0xC4 || op == 0xC5 || op == 0xC6)
        imm = fetch8(d);
    if (!finish(s, d)) return d->addr16 ? X64_EXIT_UNSUPPORTED : X64_EXIT_ILLEGAL;
    s->rip = d->next;
    return sse_body(s, d, op, pfx, imm);
}

// The 0F 10-17, 28-2F, 50-5F, C2 and C4-C6 rows, and 60-7F and D0-FF with a
// prefix, once decoded (d finished, rip after the instruction): also run from
// the decoded-instruction cache.
static int sse_body(X64State *s, Dec *d, uint8_t op, int pfx, uint8_t imm)
{
    int i;
    bool sse_int = (op >= 0x60 && op <= 0x7F) || op >= 0xD0 || op == 0xC4 || op == 0xC5;
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
    case 0x2A: { // CVTSI2SS CVTSI2SD; CVTPI2PS CVTPI2PD
        if (pfx < 2) {
            uint64_t src = d->mem ? mem_read(d->ea, 8) : mm_get(s, d->rmf);
            if (!d->mem) mmx_enter(s);
            int32_t lo = (int32_t)src, hi = (int32_t)(src >> 32);
            if (pfx == 0) {
                volatile float f0 = (float)lo, f1 = (float)hi;
                a->d[0] = f2u(f0); a->d[1] = f2u(f1);
            } else {
                a->q[0] = d2u((double)lo); a->q[1] = d2u((double)hi);
            }
            return CONT;
        }
        int size = (d->rex & 8) ? 8 : 4;
        int64_t v = sext(get_rm(s, d, size), size);
        if (pfx == 2) { volatile float f = (float)v; a->d[0] = f2u(f); }
        else { volatile double f = (double)v; a->q[0] = d2u(f); }
        return CONT;
    }
    case 0x2C: case 0x2D: { // CVTTSS2SI CVTSS2SI CVTTSD2SI CVTSD2SI; CVT(T)PS2PI CVT(T)PD2PI
        if (pfx < 2) {
            double v0, v1;
            if (pfx == 0) { b = xm(s, d, 8); v0 = u2f(f32_in(s, b.d[0])); v1 = u2f(f32_in(s, b.d[1])); }
            else { b = xm(s, d, 16); v0 = u2d(f64_in(s, b.q[0])); v1 = u2d(f64_in(s, b.q[1])); }
            mmx_enter(s);
            mm_set(s, d->regf, (uint32_t)f_to_int(v0, op == 0x2C, 4) | ((uint64_t)(uint32_t)f_to_int(v1, op == 0x2C, 4) << 32));
            return CONT;
        }
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
    case 0xD6: // MOVQ xmm/m64, xmm; MOVQ2DQ; MOVDQ2Q
        if (pfx >= 2) {
            if (d->mem) return X64_EXIT_ILLEGAL;
            mmx_enter(s);
            if (pfx == 2) { a->q[0] = mm_get(s, d->rmf); a->q[1] = 0; }
            else mm_set(s, d->regf, s->xmm[d->rm].q[0]);
            return CONT;
        }
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
        if (d->m32 && d->addrsize) return X64_EXIT_UNSUPPORTED;   // DI
        uint64_t base = (d->addrsize || d->m32 ? s->gpr[X64_RDI] & 0xffffffffull : s->gpr[X64_RDI]) + seg_base(s, d);
        for (i = 0; i < 16; i++)
            if (s->xmm[d->rm].b[i] & 0x80) mem_write(ha(s, base + (uint64_t)i), 1, a->b[i]);
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

// A string instruction's operand: the register as wide as the address size, plus
// the segment base, as a host address.
static inline uint64_t str_addr(const X64State *s, uint64_t reg, uint64_t amask, uint64_t segbase)
{
    uint64_t a = (reg & amask) + segbase;
    return s->mode32 ? s->guest_base + (uint32_t)a : a;
}

static int string_op(X64State *s, Dec *d, uint8_t op)
{
    int size = (op & 1) ? osz(d) : 1;
    bool a32 = d->addrsize || s->mode32;
    uint64_t amask = a32 ? 0xffffffffull : ~0ull;
    int asize = a32 ? 4 : 8;
    int64_t step = (s->rflags & FL_DF) ? -size : size;
    uint64_t src_base = seg_base(s, d);
    bool rep = d->rep != 0;
    uint64_t count = rep ? s->gpr[X64_RCX] & amask : 1;
    uint64_t rsi = s->gpr[X64_RSI] & amask, rdi = s->gpr[X64_RDI] & amask;
    uint64_t done = 0;
    // A forward REP over a range that does not wrap the address size: one memmove or memset.
    uint64_t bytes = count * (uint64_t)size;
    bool linear = rep && step > 0 && count && (!a32 || (s->mode32 && ((rsi + src_base) & amask) + bytes <= 0x100000000ull
                                                       && rdi + bytes <= 0x100000000ull));

    switch (op) {
    case 0xA4: case 0xA5: { // MOVS
        uint64_t src = str_addr(s, rsi, amask, src_base), dst = str_addr(s, rdi, amask, 0);
        if (linear && (dst <= src || dst >= src + bytes)) {
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
                mem_write(str_addr(s, rdi, amask, 0), size, mem_read(str_addr(s, rsi, amask, src_base), size));
                rsi = (rsi + (uint64_t)step) & amask;
                rdi = (rdi + (uint64_t)step) & amask;
            }
        }
        break;
    }
    case 0xAA: case 0xAB: { // STOS
        uint64_t v = s->gpr[X64_RAX] & mask_of(size);
        if (linear && size == 1) {
            memset((void *)(uintptr_t)str_addr(s, rdi, amask, 0), (int)v, (size_t)count);
            rdi += count; done = count;
        } else {
            for (; done < count; done++) {
                mem_write(str_addr(s, rdi, amask, 0), size, v);
                rdi = (rdi + (uint64_t)step) & amask;
            }
        }
        break;
    }
    case 0xAC: case 0xAD: // LODS
        for (; done < count; done++) {
            set_reg(s, d, X64_RAX, size, mem_read(str_addr(s, rsi, amask, src_base), size));
            rsi = (rsi + (uint64_t)step) & amask;
        }
        break;
    case 0xA6: case 0xA7: case 0xAE: case 0xAF: // CMPS, SCAS
        while (done < count) {
            uint64_t a, b;
            if (op <= 0xA7) {
                a = mem_read(str_addr(s, rsi, amask, src_base), size);
                b = mem_read(str_addr(s, rdi, amask, 0), size);
                rsi = (rsi + (uint64_t)step) & amask;
            } else {
                a = s->gpr[X64_RAX];
                b = mem_read(str_addr(s, rdi, amask, 0), size);
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
    s->rip = ipm(s, target);
    return native(env, s->rip) ? X64_EXIT_NATIVE : CONT;
}

static int call(X64State *s, const X64Env *env, uint64_t target, uint64_t ret)
{
    push(s, ret, s->mode32 ? 4 : 8);
    return jump(s, env, target);
}

static uint64_t rdtsc(void)
{
    return host_nanoseconds();
}

// FXSAVE's 512-byte area: control and status words, the abridged tag byte,
// MXCSR, the x87 registers in stack order and the XMM registers (8 of them in
// 32-bit mode). Bytes 416-511 are left alone, as x86 leaves 464-511 to software.
// The instruction and operand pointers are not tracked and are stored as zero.
static void fx_store(const X64State *s, uint8_t *area)
{
    memset(area, 0, 416);
    memcpy(area + 0, &s->fcw, 2);
    memcpy(area + 2, &s->fsw, 2);
    area[4] = s->ftw;
    memcpy(area + 24, &s->mxcsr, 4);
    uint32_t mask = 0xffff;
    memcpy(area + 28, &mask, 4);
    int top = (s->fsw >> 11) & 7;
    for (int i = 0; i < 8; i++) memcpy(area + 32 + 16 * i, &s->st[(top + i) & 7], 10);
    memcpy(area + 160, s->xmm, s->mode32 ? 8 * sizeof s->xmm[0] : sizeof s->xmm);
}

static void fx_load(X64State *s, const uint8_t *area)
{
    memcpy(&s->fcw, area + 0, 2);
    memcpy(&s->fsw, area + 2, 2);
    s->ftw = area[4];
    memcpy(&s->mxcsr, area + 24, 4);
    s->mxcsr &= 0xffff;
    int top = (s->fsw >> 11) & 7;
    for (int i = 0; i < 8; i++) {
        X87Reg *r = &s->st[(top + i) & 7];
        memcpy(r, area + 32 + 16 * i, 10);
        r->pad[0] = r->pad[1] = r->pad[2] = 0;
    }
    memcpy(s->xmm, area + 160, s->mode32 ? 8 * sizeof s->xmm[0] : sizeof s->xmm);
}

static void fxsave(const X64State *s, uint64_t addr)
{
    uint8_t area[416];
    fx_store(s, area);
    memcpy((void *)(uintptr_t)addr, area, sizeof area);
}

static void fxrstor(X64State *s, uint64_t addr)
{
    uint8_t area[416];
    memcpy(area, (const void *)(uintptr_t)addr, sizeof area);
    fx_load(s, area);
    set_mxcsr(s, s->mxcsr);
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

#define FIN() do { if (!finish(s, d)) return d->addr16 ? X64_EXIT_UNSUPPORTED : X64_EXIT_ILLEGAL; s->rip = d->next; } while (0)
#define NEED_MEM() do { if (!d->mem) return X64_EXIT_ILLEGAL; } while (0)
#define LOCK_MEM() do { if (d->lock && !d->mem) return X64_EXIT_ILLEGAL; } while (0)

// ---------------------------------------------------------------------------
// x87 and MMX
//
// Registers keep all 80 bits through loads, stores, moves, integer
// conversions and comparisons. FADD, FSUB, FMUL, FDIV and FSQRT round as the
// control word says, to 24, 53 or 64 bits (precision control) in the rounding
// control's direction, with x87's exponent range: exactly, as the x87 does
// (f80_arith). Transcendental results are double's. Exceptions are not raised
// and their flags are not set.

#define FSW_C0 0x0100
#define FSW_C1 0x0200
#define FSW_C2 0x0400
#define FSW_C3 0x4000
#define FSW_CC (FSW_C0 | FSW_C1 | FSW_C2 | FSW_C3)

static const X87Reg F80_INDEFINITE = { 0xc000000000000000ull, 0xffff, { 0, 0, 0 } };
static const X87Reg F80_ONE = { 0x8000000000000000ull, 0x3fff, { 0, 0, 0 } };

static inline int x87_top(const X64State *s) { return (s->fsw >> 11) & 7; }
static inline void x87_set_top(X64State *s, int top) { s->fsw = (uint16_t)((s->fsw & ~0x3800) | ((top & 7) << 11)); }
static inline X87Reg *x87_st(X64State *s, int i) { return &s->st[(x87_top(s) + i) & 7]; }
static inline void x87_set_cc(X64State *s, uint16_t cc) { s->fsw = (uint16_t)((s->fsw & ~FSW_CC) | cc); }

// A push onto a register still in use is a (masked) stack overflow: the
// indefinite NaN is pushed instead.
static inline void x87_push(X64State *s, X87Reg v)
{
    int t = (x87_top(s) - 1) & 7;
    x87_set_top(s, t);
    s->st[t] = (s->ftw & (1 << t)) ? F80_INDEFINITE : v;
    s->ftw |= (uint8_t)(1 << t);
}

static inline bool x87_valid(const X64State *s, int i) { return (s->ftw >> ((x87_top(s) + i) & 7)) & 1; }

// Reading an empty register is a (masked) stack underflow: the operand is the
// indefinite NaN.
static inline X87Reg x87_get(X64State *s, int i) { return x87_valid(s, i) ? *x87_st(s, i) : F80_INDEFINITE; }

static inline void x87_put(X64State *s, int i, X87Reg v)
{
    int p = (x87_top(s) + i) & 7;
    s->st[p] = v;
    s->ftw |= (uint8_t)(1 << p);
}

static inline void x87_pop(X64State *s)
{
    int t = x87_top(s);
    s->ftw &= (uint8_t)~(1 << t);
    x87_set_top(s, t + 1);
}

static inline bool f80_nan(X87Reg v) { return (v.exponent & 0x7fff) == 0x7fff && (v.mantissa >> 63) && (v.mantissa << 1); }
// Encodings the x87 has refused as operands since the 387 (an invalid operation):
// pseudo-NaNs and pseudo-infinities (exponent all ones, no integer bit) and unnormals.
static inline bool f80_unsupported(X87Reg v)
{
    int e = v.exponent & 0x7fff;
    return e && !(v.mantissa >> 63);
}
static inline bool f80_neg(X87Reg v) { return (v.exponent & 0x8000) != 0; }
static inline X87Reg f80_quiet(X87Reg v) { v.mantissa |= 0x4000000000000000ull; return v; }

static X87Reg f80_from_f64(uint64_t bits)
{
    X87Reg r = { 0, 0, { 0, 0, 0 } };
    uint16_t sign = (uint16_t)((bits >> 48) & 0x8000);
    int e = (int)((bits >> 52) & 0x7ff);
    uint64_t frac = bits & 0x000fffffffffffffull;
    if (e == 0x7ff) { r.exponent = sign | 0x7fff; r.mantissa = 0x8000000000000000ull | (frac << 11); }
    else if (e) { r.exponent = (uint16_t)(sign | (e - 1023 + 16383)); r.mantissa = 0x8000000000000000ull | (frac << 11); }
    else if (frac) { int lz = __builtin_clzll(frac); r.mantissa = frac << lz; r.exponent = (uint16_t)(sign | (15372 - lz)); }
    else r.exponent = sign;
    return r;
}

static X87Reg f80_from_f32(uint32_t bits)
{
    X87Reg r = { 0, 0, { 0, 0, 0 } };
    uint16_t sign = (uint16_t)((bits >> 16) & 0x8000);
    int e = (int)((bits >> 23) & 0xff);
    uint64_t frac = bits & 0x007fffffu;
    if (e == 0xff) { r.exponent = sign | 0x7fff; r.mantissa = 0x8000000000000000ull | (frac << 40); }
    else if (e) { r.exponent = (uint16_t)(sign | (e - 127 + 16383)); r.mantissa = 0x8000000000000000ull | (frac << 40); }
    else if (frac) { int lz = __builtin_clzll(frac); r.mantissa = frac << lz; r.exponent = (uint16_t)(sign | (16297 - lz)); }
    else r.exponent = sign;
    return r;
}

static X87Reg f80_from_magnitude(bool neg, uint64_t m)
{
    X87Reg r = { 0, (uint16_t)(neg ? 0x8000 : 0), { 0, 0, 0 } };
    if (!m) return r;
    int lz = __builtin_clzll(m);
    r.mantissa = m << lz;
    r.exponent = (uint16_t)(r.exponent | (16383 + 63 - lz));
    return r;
}

static X87Reg f80_from_int(int64_t i)
{
    return f80_from_magnitude(i < 0, i < 0 ? 0 - (uint64_t)i : (uint64_t)i);
}

// v's bits as a double, when a normal double holds it exactly.
static inline bool f80_double_bits(X87Reg v, uint64_t *bits)
{
    unsigned e = v.exponent & 0x7fff;
    if (e < 16383 - 1022 || e > 16383 + 1023 || !(v.mantissa >> 63) || (v.mantissa & 0x7ff)) return false;
    *bits = ((uint64_t)(v.exponent & 0x8000) << 48) | ((uint64_t)(e - 16383 + 1023) << 52) | ((v.mantissa >> 11) & 0x000fffffffffffffull);
    return true;
}

typedef unsigned __int128 u128;

// v as an IEEE number with `frac` fraction bits and `ebits` exponent bits (float:
// 23 and 8, double: 52 and 11), rounded once, in the host's rounding direction
// (x87() sets it from the control word), to a normal or denormal result: what
// FST m32 and FST m64 store.
static uint64_t f80_to_ieee(X87Reg v, int frac, int ebits)
{
    int e = v.exponent & 0x7fff;
    bool neg = f80_neg(v);
    uint64_t sign = (uint64_t)neg << (frac + ebits), ones = (1ull << ebits) - 1, fmask = (1ull << frac) - 1;
    if (f80_unsupported(v)) return (1ull << (frac + ebits)) | (ones << frac) | (1ull << (frac - 1));   // the default NaN
    if (e == 0x7fff) {
        if (!(v.mantissa << 1)) return sign | (ones << frac);                                           // infinity
        return sign | (ones << frac) | (1ull << (frac - 1)) | ((v.mantissa >> (63 - frac)) & fmask);    // a NaN, quieted
    }
    if (!v.mantissa) return sign;
    int lz = __builtin_clzll(v.mantissa);
    uint64_t m = v.mantissa << lz;
    int exp = (e ? e : 1) - 16383 - lz;               // of m's top bit
    int bias = (1 << (ebits - 1)) - 1, emin = 1 - bias;
    int shift = 63 - frac;                            // the bits a normal result drops
    if (exp < emin) shift += emin - exp;              // a denormal keeps fewer
    if (shift > 126) shift = 126;
    u128 unit = (u128)1 << shift, low = (u128)m & (unit - 1), half = unit >> 1;
    uint64_t q = (uint64_t)((u128)m >> shift);
    bool up;
    switch (host_get_rc()) {
    case 0: up = low > half || (low == half && (q & 1)); break;
    case 1: up = neg && low; break;
    case 2: up = !neg && low; break;
    default: up = false; break;
    }
    q += up;
    if (exp < emin) return sign | ((q >> frac) << frac) | (q & fmask);   // rounding up to the smallest normal sets its exponent
    if (q >> (frac + 1)) { q >>= 1; exp++; }
    if (exp > bias) {                                 // overflow: infinity, or the largest finite number
        int rc = host_get_rc();
        if (rc == 0 || (rc == 1 && neg) || (rc == 2 && !neg)) return sign | (ones << frac);
        return sign | ((ones - 1) << frac) | fmask;
    }
    return sign | ((uint64_t)(exp + bias) << frac) | (q & fmask);
}

static double f80_to_double(X87Reg v)
{
    uint64_t exact;
    if (f80_double_bits(v, &exact)) return u2d(exact);   // no rounding to do
    return u2d(f80_to_ieee(v, 52, 11));
}

static uint32_t f80_to_f32(X87Reg v)
{
    return (uint32_t)f80_to_ieee(v, 23, 8);
}

// Rounds the magnitude to an integer: rc 0 nearest even, 1 down, 2 up, 3 toward
// zero. False when it does not fit in 64 bits (or for a NaN or infinity).
static bool f80_round(X87Reg v, int rc, uint64_t *magnitude)
{
    int e = v.exponent & 0x7fff;
    bool neg = f80_neg(v);
    if (e == 0x7fff || f80_unsupported(v)) return false;
    if (!v.mantissa) { *magnitude = 0; return true; }
    int ue = (e ? e : 1) - 16383;   // value = mantissa * 2^(ue - 63)
    if (ue >= 63) {
        if (ue > 63) return false;
        *magnitude = v.mantissa;     // 2^63 to 2^64 - 1, already an integer
        return true;
    }
    uint64_t ip, frac;               // integer part, fraction left-aligned
    if (ue >= 0) {
        ip = v.mantissa >> (63 - ue);
        frac = v.mantissa << (ue + 1);
    } else {
        int shift = -ue - 1;   // the value is mantissa * 2^(ue + 1) / 2^64: below 1
        ip = 0;
        frac = shift == 0 ? v.mantissa : shift >= 64 ? 1 : (v.mantissa >> shift) | ((v.mantissa << (64 - shift)) != 0);
    }
    bool up;
    switch (rc & 3) {
    case 0: up = frac > 0x8000000000000000ull || (frac == 0x8000000000000000ull && (ip & 1)); break;
    case 1: up = neg && frac; break;
    case 2: up = !neg && frac; break;
    default: up = false; break;
    }
    *magnitude = ip + up;
    return true;
}

static bool f80_to_int(X87Reg v, int rc, int64_t *out)
{
    uint64_t m;
    if (!f80_round(v, rc, &m)) return false;
    if (f80_neg(v)) {
        if (m > 0x8000000000000000ull) return false;
        *out = (int64_t)(0 - m);
    } else {
        if (m > 0x7fffffffffffffffull) return false;
        *out = (int64_t)m;
    }
    return true;
}

// -1 less, 0 equal, 1 greater, 2 unordered; exact on all 80 bits.
static int f80_cmp(X87Reg a, X87Reg b)
{
    if (f80_nan(a) || f80_nan(b) || f80_unsupported(a) || f80_unsupported(b)) return 2;
    bool az = !a.mantissa, bz = !b.mantissa;
    bool an = f80_neg(a), bn = f80_neg(b);
    if (az && bz) return 0;
    if (az) return bn ? 1 : -1;
    if (bz) return an ? -1 : 1;
    if (an != bn) return an ? -1 : 1;
    int ea = a.exponent & 0x7fff, eb = b.exponent & 0x7fff;
    int m = ea != eb ? (ea < eb ? -1 : 1) : a.mantissa != b.mantissa ? (a.mantissa < b.mantissa ? -1 : 1) : 0;
    return an ? -m : m;
}

// --- Exact x87 arithmetic --------------------------------------------------
//
// Values that doubles hold exactly take the host's double arithmetic at 24 and
// 53 bits, when the result is a normal double too: it rounds the same way, in
// the host's rounding mode, which x87() sets from the control word, and a double
// result rounded again to 24 bits is the correctly rounded one (53 >= 2 * 24 + 2).
// Everything else, and all of 64-bit precision, takes integer arithmetic on the
// 64-bit significands, rounded once by f80_pack.

enum { F80_ZERO, F80_FINITE, F80_INF, F80_NAN, F80_BAD };

typedef struct {
    int kind;
    bool neg;
    int32_t exp2;    // F80_FINITE: the value is m * 2^exp2
    uint64_t m;
} F80Parts;

static F80Parts f80_parts(X87Reg v)
{
    F80Parts p = { F80_FINITE, f80_neg(v), 0, v.mantissa };
    int e = v.exponent & 0x7fff;
    bool integer = (v.mantissa >> 63) != 0;
    if (e == 0x7fff) p.kind = !integer ? F80_BAD : (v.mantissa << 1) ? F80_NAN : F80_INF;   // pseudo-NaN, pseudo-infinity
    else if (e && !integer) p.kind = F80_BAD;                                               // unnormal
    else if (!v.mantissa) p.kind = F80_ZERO;
    else p.exp2 = (e ? e : 1) - 16383 - 63;                                                 // denormals: as exponent 1
    return p;
}

static int x87_precision(uint16_t fcw)
{
    int pc = (fcw >> 8) & 3;
    return pc == 0 ? 24 : pc == 2 ? 53 : 64;
}

static inline int clz128(u128 v)   // v != 0
{
    uint64_t hi = (uint64_t)(v >> 64);
    return hi ? __builtin_clzll(hi) : 64 + __builtin_clzll((uint64_t)v);
}

static X87Reg f80_signed(bool neg, uint16_t exponent, uint64_t mantissa)
{
    X87Reg r = { mantissa, (uint16_t)(exponent | (neg ? 0x8000 : 0)), { 0, 0, 0 } };
    return r;
}

// sig * 2^exp2 (sig != 0; sticky: the exact value has more below sig's last
// bit), rounded to `bits` significant bits in direction rc. A tiny result is
// denormalized first and rounded at the same bit positions; an overflow gives
// infinity or the largest finite value, as rc says (both as masked exceptions do).
static X87Reg f80_pack(bool neg, int32_t exp2, u128 sig, bool sticky, int bits, int rc)
{
    int lz = clz128(sig);
    sig <<= lz;
    int32_t e = exp2 - lz + 127 + 16383;   // the biased exponent of sig's bit 127
    if (e < 1) {
        int32_t shift = 1 - e;
        if (shift >= 128) { sticky = true; sig = 0; }
        else { sticky = sticky || (sig << (128 - shift)) != 0; sig >>= shift; }
        e = 1;
    }
    u128 unit = (u128)1 << (128 - bits), low = sig & (unit - 1), half = unit >> 1;
    u128 kept = sig - low;
    bool inexact = low || sticky, up;
    switch (rc & 3) {
    case 0: up = low > half || (low == half && (sticky || (kept & unit))); break;
    case 1: up = neg && inexact; break;
    case 2: up = !neg && inexact; break;
    default: up = false; break;
    }
    if (up) {
        kept += unit;
        if (!kept) { kept = (u128)1 << 127; e++; }   // carried out of the top
    }
    if (e >= 0x7fff) {
        if (rc == 0 || (rc == 1 && neg) || (rc == 2 && !neg)) return f80_signed(neg, 0x7fff, 0x8000000000000000ull);
        return f80_signed(neg, 0x7ffe, ~0ull << (64 - bits));
    }
    uint64_t m = (uint64_t)(kept >> 64);
    return f80_signed(neg, (uint16_t)((m >> 63) ? e : 0), m);   // a denormal (or zero) below exponent 1
}

// Whether a double holds v exactly: zero, infinity, or a normal double.
static bool f80_is_double(X87Reg v)
{
    int e = v.exponent & 0x7fff;
    if (e == 0x7fff) return v.mantissa == 0x8000000000000000ull;
    if (!v.mantissa) return e == 0;
    return (v.mantissa >> 63) && !(v.mantissa & 0x7ff) && e >= 16383 - 1022 && e <= 16383 + 1023;
}

// To `bits` bits (24), in the x87's exponent range.
static X87Reg f80_to_precision(X87Reg v, int bits, int rc)
{
    F80Parts p = f80_parts(v);
    return p.kind == F80_FINITE ? f80_pack(p.neg, p.exp2, p.m, false, bits, rc) : v;
}

static uint64_t isqrt128(u128 n, u128 *remainder)
{
    u128 rem = 0;
    uint64_t root = 0;
    for (int i = 0; i < 64; i++) {
        rem = (rem << 2) | (n >> 126);
        n <<= 2;
        u128 trial = ((u128)root << 2) | 1;
        root <<= 1;
        if (rem >= trial) { rem -= trial; root |= 1; }
    }
    *remainder = rem;
    return root;
}

static X87Reg f80_sqrt(X87Reg a, uint16_t fcw)
{
    F80Parts x = f80_parts(a);
    int bits = x87_precision(fcw), rc = (fcw >> 10) & 3;
    switch (x.kind) {
    case F80_BAD: return F80_INDEFINITE;
    case F80_NAN: return f80_quiet(a);
    case F80_ZERO: return a;                                  // -0 too
    case F80_INF: return x.neg ? F80_INDEFINITE : a;
    }
    if (x.neg) return F80_INDEFINITE;
    if (bits != 64 && f80_is_double(a)) {
        volatile double r = sqrt(f80_to_double(a));            // never tiny, never overflows
        X87Reg v = f80_from_f64(d2u(r));
        return bits == 24 ? f80_to_precision(v, 24, rc) : v;
    }
    int lz = __builtin_clzll(x.m);
    uint64_t m = x.m << lz;
    int32_t e = x.exp2 - lz;
    int k = (e & 1) ? 63 : 64;                                // an even exponent left over
    u128 n = (u128)m << k, rem;
    uint64_t root = isqrt128(n, &rem);
    // What is below the root's last bit: above a half when rem > root ((root + 1/2)^2 = root^2 + root + 1/4).
    u128 sig = ((u128)root << 64) | (rem > root ? ((u128)1 << 63) | 1 : rem ? 1 : 0);
    return f80_pack(false, (e - k) / 2 - 64, sig, false, bits, rc);
}

enum { X87_ADD, X87_MUL, X87_COM, X87_COMP, X87_SUB, X87_SUBR, X87_DIV, X87_DIVR };

static X87Reg f80_arith_full(int op, X87Reg a, X87Reg b, uint16_t fcw);

// `a` is the destination: SUB is a - b, SUBR b - a.
static inline X87Reg f80_arith(int op, X87Reg a, X87Reg b, uint16_t fcw)
{
    // The common case first: 53-bit precision (or 24), two normal doubles, a result strictly
    // inside the normal doubles. Rounded by the host in the control word's direction (x87()).
    uint64_t pa, pb;
    if (((fcw & 0x300) == 0x200 || (fcw & 0x300) == 0) && f80_double_bits(a, &pa) && f80_double_bits(b, &pb)) {
        double p = u2d(pa), q = u2d(pb);
        volatile double r;
        switch (op) {
        case X87_ADD: r = p + q; break;
        case X87_MUL: r = p * q; break;
        case X87_SUB: r = p - q; break;
        case X87_SUBR: r = q - p; break;
        case X87_DIV: r = p / q; break;
        default: r = q / p; break;
        }
        uint64_t rb = d2u(r), magnitude = rb & 0x7fffffffffffffffull;
        if (magnitude > 0x0010000000000000ull && magnitude < 0x7fefffffffffffffull) {
            X87Reg v = { 0x8000000000000000ull | ((rb & 0x000fffffffffffffull) << 11),
                         (uint16_t)(((rb >> 48) & 0x8000) | (((rb >> 52) & 0x7ff) - 1023 + 16383)), { 0, 0, 0 } };
            return (fcw & 0x300) ? v : f80_to_precision(v, 24, (fcw >> 10) & 3);
        }
    }
    return f80_arith_full(op, a, b, fcw);
}

// Everything else: exactly, at any precision (f80_pack).
static __attribute__((noinline)) X87Reg f80_arith_full(int op, X87Reg a, X87Reg b, uint16_t fcw)
{
    F80Parts x = f80_parts(a), y = f80_parts(b);
    if (x.kind == F80_BAD || y.kind == F80_BAD) return F80_INDEFINITE;
    bool na = f80_nan(a), nb = f80_nan(b);
    if (na || nb) {
        if (na && nb) {
            bool qa = (a.mantissa >> 62) & 1, qb = (b.mantissa >> 62) & 1;
            if (qa != qb) return qa ? a : b;   // a QNaN over an SNaN
            uint64_t ma = a.mantissa << 1, mb = b.mantissa << 1;
            if (ma != mb) return f80_quiet(ma > mb ? a : b);   // the larger significand
            return f80_quiet(f80_neg(a) ? b : a);              // equal ones: the positive NaN
        }
        return f80_quiet(na ? a : b);
    }
    if (op == X87_SUBR || op == X87_DIVR) {
        F80Parts t = x; x = y; y = t;
        X87Reg u = a; a = b; b = u;
        op = op == X87_SUBR ? X87_SUB : X87_DIV;
    }
    int bits = x87_precision(fcw), rc = (fcw >> 10) & 3;

    if (bits != 64 && f80_is_double(a) && f80_is_double(b)) {
        double p = f80_to_double(a), q = f80_to_double(b);
        volatile double r = op == X87_ADD ? p + q : op == X87_SUB ? p - q : op == X87_MUL ? p * q : p / q;
        uint64_t rb = d2u(r);
        if (f64_nan(rb)) return F80_INDEFINITE;               // inf - inf, 0 * inf, 0 / 0, inf / inf
        // A double result strictly between the smallest and the largest normal double is the
        // x87's (rounding is monotonic: the exact value is in that range too), and so is a zero or
        // an infinity that is exact (a sum is zero only exactly); others go the long way.
        bool special = x.kind != F80_FINITE || y.kind != F80_FINITE;
        bool normal = fabs(r) > 0x1p-1022 && fabs(r) < 0x1.fffffffffffffp1023;
        bool exact_zero = r == 0 && (special || op == X87_ADD || op == X87_SUB);
        bool exact_inf = isinf(r) && (x.kind == F80_INF || y.kind == F80_INF || (op == X87_DIV && y.kind == F80_ZERO));
        if (normal || exact_zero || exact_inf) {
            X87Reg v = f80_from_f64(rb);
            return bits == 24 ? f80_to_precision(v, 24, rc) : v;
        }
    }

    bool neg;
    switch (op) {
    case X87_ADD: case X87_SUB: {
        bool yneg = y.neg != (op == X87_SUB);
        if (x.kind == F80_INF || y.kind == F80_INF) {
            if (x.kind == F80_INF && y.kind == F80_INF && x.neg != yneg) return F80_INDEFINITE;
            return f80_signed(x.kind == F80_INF ? x.neg : yneg, 0x7fff, 0x8000000000000000ull);
        }
        if (x.kind == F80_ZERO && y.kind == F80_ZERO) return f80_signed(x.neg == yneg ? x.neg : rc == 1, 0, 0);
        if (y.kind == F80_ZERO) return f80_pack(x.neg, x.exp2, x.m, false, bits, rc);
        if (x.kind == F80_ZERO) return f80_pack(yneg, y.exp2, y.m, false, bits, rc);
        // The larger magnitude first: a larger exponent is a normal number, larger than any with a smaller one.
        bool bneg = x.neg, sneg = yneg;
        F80Parts big = x, small = y;
        if (y.exp2 > x.exp2 || (y.exp2 == x.exp2 && y.m > x.m)) { big = y; small = x; bneg = yneg; sneg = x.neg; }
        int32_t d = big.exp2 - small.exp2;
        u128 A = (u128)big.m << 62, B = (u128)small.m << 62;
        bool sticky = false;
        if (d >= 128) { sticky = true; B = 0; }
        else if (d) { sticky = (B << (128 - d)) != 0; B >>= d; }
        u128 sum;
        if (bneg == sneg) sum = A + B;
        else sum = A - B - (sticky ? 1 : 0);   // with a sticky bit, the difference is just below A - B
        if (!sum && !sticky) return f80_signed(rc == 1, 0, 0);
        return f80_pack(bneg, big.exp2 - 62, sum, sticky, bits, rc);
    }
    case X87_MUL:
        neg = x.neg != y.neg;
        if (x.kind == F80_INF || y.kind == F80_INF)
            return x.kind == F80_ZERO || y.kind == F80_ZERO ? F80_INDEFINITE : f80_signed(neg, 0x7fff, 0x8000000000000000ull);
        if (x.kind == F80_ZERO || y.kind == F80_ZERO) return f80_signed(neg, 0, 0);
        return f80_pack(neg, x.exp2 + y.exp2, (u128)x.m * y.m, false, bits, rc);
    default: {   // X87_DIV
        neg = x.neg != y.neg;
        if (x.kind == F80_INF) return y.kind == F80_INF ? F80_INDEFINITE : f80_signed(neg, 0x7fff, 0x8000000000000000ull);
        if (y.kind == F80_INF) return f80_signed(neg, 0, 0);
        if (y.kind == F80_ZERO) return x.kind == F80_ZERO ? F80_INDEFINITE : f80_signed(neg, 0x7fff, 0x8000000000000000ull);
        if (x.kind == F80_ZERO) return f80_signed(neg, 0, 0);
        int lx = __builtin_clzll(x.m), ly = __builtin_clzll(y.m);
        uint64_t mx = x.m << lx, my = y.m << ly;
        // 66 quotient bits of mx / my (in [1/2, 2)): 64 significant, one more, and the remainder.
        u128 r = mx, q = 0;
        for (int i = 0; i < 66; i++) {
            q <<= 1;
            if (r >= my) { r -= my; q |= 1; }
            r <<= 1;
        }
        return f80_pack(neg, (x.exp2 - lx) - (y.exp2 - ly) - 65, q, r != 0, bits, rc);
    }
    }
}

static X87Reg f80_from_double(double r)
{
    uint64_t bits = d2u(r);
    return f64_nan(bits) ? F80_INDEFINITE : f80_from_f64(bits);
}

static void x87_compare(X64State *s, X87Reg a, X87Reg b, bool eflags)
{
    int c = f80_cmp(a, b);
    if (eflags) {
        set_flags(s, FL_ARITH, c == 2 ? FL_ZF | FL_PF | FL_CF : c < 0 ? FL_CF : c == 0 ? FL_ZF : 0);
        s->fsw &= (uint16_t)~FSW_C1;
    } else {
        x87_set_cc(s, c == 2 ? FSW_C3 | FSW_C2 | FSW_C0 : c < 0 ? FSW_C0 : c == 0 ? FSW_C3 : 0);
    }
}

// The full two-bit tag word FSTENV and FSAVE store.
static uint16_t x87_full_tags(const X64State *s)
{
    uint16_t tags = 0;
    for (int i = 0; i < 8; i++) {
        unsigned tag;
        const X87Reg *r = &s->st[i];
        int e = r->exponent & 0x7fff;
        if (!(s->ftw & (1 << i))) tag = 3;
        else if (e == 0x7fff) tag = 2;
        else if (!e) tag = r->mantissa ? 2 : 1;
        else tag = (r->mantissa >> 63) ? 0 : 2;
        tags |= (uint16_t)(tag << (2 * i));
    }
    return tags;
}

// FSTENV's 28-byte protected-mode image. The instruction and operand pointers
// are not tracked and are stored as zero.
static void x87_store_env(X64State *s, uint64_t addr)
{
    uint32_t env[7] = { 0xffff0000u | s->fcw, 0xffff0000u | s->fsw, 0xffff0000u | x87_full_tags(s), 0, 0, 0, 0xffff0000u };
    memcpy((void *)(uintptr_t)addr, env, sizeof env);
}

static void x87_load_env(X64State *s, uint64_t addr)
{
    uint32_t env[7];
    memcpy(env, (const void *)(uintptr_t)addr, sizeof env);
    s->fcw = (uint16_t)env[0];
    s->fsw = (uint16_t)env[1];
    s->ftw = 0;
    for (int i = 0; i < 8; i++)
        if (((env[2] >> (2 * i)) & 3) != 3) s->ftw |= (uint8_t)(1 << i);
}

static void x87_init(X64State *s)
{
    s->fcw = 0x37f;
    s->fsw = 0;
    s->ftw = 0;
}

// The constants, rounded as RC asks: { nearest, down/toward zero, up } significands.
static X87Reg x87_constant(const X64State *s, int which)
{
    static const struct { uint16_t exponent; uint64_t near, down, up; } k[] = {
        { 0x4000, 0xd49a784bcd1b8afeull, 0xd49a784bcd1b8afeull, 0xd49a784bcd1b8affull },   // log2(10)
        { 0x3fff, 0xb8aa3b295c17f0bcull, 0xb8aa3b295c17f0bbull, 0xb8aa3b295c17f0bcull },   // log2(e)
        { 0x4000, 0xc90fdaa22168c235ull, 0xc90fdaa22168c234ull, 0xc90fdaa22168c235ull },   // pi
        { 0x3ffd, 0x9a209a84fbcff799ull, 0x9a209a84fbcff798ull, 0x9a209a84fbcff799ull },   // log10(2)
        { 0x3ffe, 0xb17217f7d1cf79acull, 0xb17217f7d1cf79abull, 0xb17217f7d1cf79acull },   // ln(2)
    };
    int rc = (s->fcw >> 10) & 3;
    X87Reg r = { rc == 0 ? k[which].near : rc == 2 ? k[which].up : k[which].down, k[which].exponent, { 0, 0, 0 } };
    return r;
}

static X87Reg x87_load_bcd(uint64_t addr)
{
    uint8_t b[10];
    memcpy(b, (const void *)(uintptr_t)addr, 10);
    uint64_t v = 0;
    for (int i = 8; i >= 0; i--) v = v * 100 + (uint64_t)(b[i] >> 4) * 10 + (b[i] & 15);
    return f80_from_magnitude((b[9] & 0x80) != 0, v);
}

static void x87_store_bcd(const X64State *s, uint64_t addr, X87Reg v)
{
    uint8_t b[10] = { 0 };
    uint64_t m;
    if (!f80_round(v, (s->fcw >> 10) & 3, &m) || m > 999999999999999999ull) {
        b[7] = 0xc0; b[8] = 0xff; b[9] = 0xff;   // the packed BCD indefinite
    } else {
        for (int i = 0; i < 9; i++) { b[i] = (uint8_t)((m % 10) | ((m / 10 % 10) << 4)); m /= 100; }
        if (f80_neg(v)) b[9] = 0x80;
    }
    memcpy((void *)(uintptr_t)addr, b, 10);
}

// D9 E0-FF and the rest of the register-only one-operand group.
static int x87_misc(X64State *s, uint8_t low)
{
    X87Reg a = x87_get(s, 0), b = x87_get(s, 1);
    double x = f80_to_double(a);
    // An invalid operand (an unsupported encoding, or an empty register for an instruction with
    // two) makes the result the default NaN, as the masked invalid-operation response does.
    bool one = low == 0xF0 || low == 0xF2 || low == 0xF4 || low == 0xFA || low == 0xFB || low == 0xFC || low == 0xFE || low == 0xFF;
    bool two = low == 0xF1 || low == 0xF3 || low == 0xF5 || low == 0xF8 || low == 0xF9 || low == 0xFD;
    if ((one && f80_unsupported(a)) ||
        (two && (f80_unsupported(a) || f80_unsupported(b) || !x87_valid(s, 0) || !x87_valid(s, 1)))) {
        switch (low) {
        case 0xF1: case 0xF3: case 0xF9: x87_put(s, 1, F80_INDEFINITE); x87_pop(s); break;   // FYL2X FPATAN FYL2XP1
        case 0xF2: case 0xF4: case 0xFB: x87_put(s, 0, F80_INDEFINITE); x87_push(s, F80_INDEFINITE); break;   // FPTAN FXTRACT FSINCOS
        default: x87_put(s, 0, F80_INDEFINITE); break;
        }
        if (low != 0xF4 && low != 0xFA && low != 0xFC && low != 0xFD) x87_set_cc(s, 0);
        return CONT;
    }
    // A NaN operand comes back quieted (two: as the arithmetic instructions choose).
    switch (low) {
    case 0xF0: case 0xFA: case 0xFE: case 0xFF:                               // F2XM1 FSQRT FSIN FCOS
        if (f80_nan(a)) { x87_put(s, 0, f80_quiet(a)); if (low != 0xFA) x87_set_cc(s, 0); return CONT; }
        break;
    case 0xF1: case 0xF3: case 0xF9:                                          // FYL2X FPATAN FYL2XP1
        if (f80_nan(a) || f80_nan(b)) { x87_put(s, 1, f80_arith(X87_ADD, a, b, s->fcw)); x87_pop(s); return CONT; }
        break;
    }
    switch (low) {
    case 0xE0: if (x87_valid(s, 0)) a.exponent ^= 0x8000; x87_put(s, 0, a); return CONT;   // FCHS
    case 0xE1: if (x87_valid(s, 0)) a.exponent &= 0x7fff; x87_put(s, 0, a); return CONT;   // FABS
    case 0xE4: x87_compare(s, a, (X87Reg){ 0, 0, { 0, 0, 0 } }, false); return CONT;   // FTST
    case 0xE5: {                                                              // FXAM
        a = *x87_st(s, 0);   // an empty register's sign is still reported
        int e = a.exponent & 0x7fff;
        uint16_t cc;
        if (!(s->ftw & (1 << x87_top(s)))) cc = FSW_C3 | FSW_C0;
        else if (e == 0x7fff) cc = !(a.mantissa >> 63) ? 0 : (a.mantissa << 1) ? FSW_C0 : FSW_C2 | FSW_C0;
        else if (!e) cc = a.mantissa ? FSW_C3 | FSW_C2 : FSW_C3;
        else cc = (a.mantissa >> 63) ? FSW_C2 : 0;
        if (f80_neg(a)) cc |= FSW_C1;
        x87_set_cc(s, cc);
        return CONT;
    }
    case 0xE8: x87_push(s, F80_ONE); return CONT;                             // FLD1
    case 0xE9: x87_push(s, x87_constant(s, 0)); return CONT;                  // FLDL2T
    case 0xEA: x87_push(s, x87_constant(s, 1)); return CONT;                  // FLDL2E
    case 0xEB: x87_push(s, x87_constant(s, 2)); return CONT;                  // FLDPI
    case 0xEC: x87_push(s, x87_constant(s, 3)); return CONT;                  // FLDLG2
    case 0xED: x87_push(s, x87_constant(s, 4)); return CONT;                  // FLDLN2
    case 0xEE: x87_push(s, (X87Reg){ 0, 0, { 0, 0, 0 } }); return CONT;      // FLDZ
    case 0xF0: x87_put(s, 0, f80_from_double(x64m_exp2m1(x))); return CONT;   // F2XM1
    case 0xF1: x87_put(s, 1, f80_from_double(f80_to_double(b) * (log(x) / LN2))); x87_pop(s); return CONT;   // FYL2X
    case 0xF2: case 0xFE: case 0xFF: case 0xFB:                               // FPTAN FSIN FCOS FSINCOS
        if (fabs(x) >= 0x1p63 && !isnan(x)) { x87_set_cc(s, FSW_C2); return CONT; }
        x87_set_cc(s, 0);
        if (low == 0xF2) { x87_put(s, 0, f80_from_double(tan(x))); x87_push(s, F80_ONE); }
        else if (low == 0xFE) x87_put(s, 0, f80_from_double(sin(x)));
        else if (low == 0xFF) x87_put(s, 0, f80_from_double(cos(x)));
        else { x87_put(s, 0, f80_from_double(sin(x))); x87_push(s, f80_from_double(cos(x))); }
        return CONT;
    case 0xF3: x87_put(s, 1, f80_from_double(atan2(f80_to_double(b), x))); x87_pop(s); return CONT;       // FPATAN
    case 0xF4: {                                                              // FXTRACT
        X87Reg exp, sig = a;
        int e = a.exponent & 0x7fff;
        if (f80_nan(a)) { exp = sig = f80_quiet(a); }
        else if (e == 0x7fff) { exp = a; exp.exponent &= 0x7fff; }
        else if (!a.mantissa) exp = f80_from_double(-INFINITY);
        else {
            int ue = (e ? e : 1) - 16383;
            if (!(sig.mantissa >> 63)) { int lz = __builtin_clzll(sig.mantissa); sig.mantissa <<= lz; ue -= lz; }
            sig.exponent = (uint16_t)((a.exponent & 0x8000) | 16383);
            exp = f80_from_int(ue);
        }
        x87_put(s, 0, exp);
        x87_push(s, sig);
        return CONT;
    }
    case 0xF5: case 0xF8: {                                                   // FPREM1 FPREM
        double y = f80_to_double(b);
        if (isnan(x) || isnan(y) || isinf(x) || y == 0) { x87_put(s, 0, isnan(x) || isnan(y) ? f80_from_double(x + y) : F80_INDEFINITE); x87_set_cc(s, 0); return CONT; }
        double r = low == 0xF8 ? x64m_fmod(x, y) : x64m_remainder(x, y);
        double q = fabs(x64m_nearbyint((x - r) / y));
        unsigned bits = (unsigned)x64m_fmod(q, 8.0);
        x87_set_cc(s, (uint16_t)(((bits & 4) ? FSW_C0 : 0) | ((bits & 2) ? FSW_C3 : 0) | ((bits & 1) ? FSW_C1 : 0)));
        x87_put(s, 0, r == 0 ? (X87Reg){ 0, (uint16_t)(a.exponent & 0x8000), { 0, 0, 0 } } : f80_from_double(r));
        return CONT;
    }
    case 0xF6: x87_set_top(s, x87_top(s) - 1); return CONT;                  // FDECSTP
    case 0xF7: x87_set_top(s, x87_top(s) + 1); return CONT;                  // FINCSTP
    case 0xF9: x87_put(s, 1, f80_from_double(f80_to_double(b) * x64m_log1p(x) / LN2)); x87_pop(s); return CONT;   // FYL2XP1
    case 0xFA: x87_put(s, 0, f80_sqrt(a, s->fcw)); return CONT;              // FSQRT
    case 0xFC: {                                                              // FRNDINT
        uint64_t m;
        int e = a.exponent & 0x7fff;
        if (e == 0x7fff || !a.mantissa || e - 16383 >= 63) { x87_put(s, 0, f80_nan(a) ? f80_quiet(a) : a); return CONT; }
        f80_round(a, (s->fcw >> 10) & 3, &m);
        x87_put(s, 0, f80_from_magnitude(f80_neg(a), m));
        return CONT;
    }
    case 0xFD: {                                                              // FSCALE
        F80Parts p = f80_parts(a), q = f80_parts(b);
        if (f80_nan(a) || f80_nan(b)) { x87_put(s, 0, f80_arith(X87_ADD, a, b, s->fcw)); return CONT; }
        if (q.kind == F80_INF) {   // by an infinity: exact, or invalid for 0 * 2^inf and inf * 2^-inf
            if ((p.kind == F80_ZERO && !q.neg) || (p.kind == F80_INF && q.neg)) x87_put(s, 0, F80_INDEFINITE);
            else if (p.kind == F80_FINITE) x87_put(s, 0, f80_signed(p.neg, q.neg ? 0 : 0x7fff, q.neg ? 0 : 0x8000000000000000ull));
            return CONT;
        }
        if (p.kind != F80_FINITE) return CONT;                               // a zero or an infinity stays
        uint64_t magnitude;                                                   // ST1 truncated, from all 80 bits
        if (!f80_round(b, 3, &magnitude) || magnitude > 100000) magnitude = 100000;   // beyond any exponent
        int32_t n = q.neg ? -(int32_t)magnitude : (int32_t)magnitude;
        // Exact unless the result is tiny or overflows (rounded at 64 bits: precision control does not apply).
        x87_put(s, 0, f80_pack(p.neg, p.exp2 + n, p.m, false, 64, (s->fcw >> 10) & 3));
        return CONT;
    }
    case 0xD0: return CONT;                                                   // FNOP
    }
    return X64_EXIT_UNSUPPORTED;
}

static int x87_body(X64State *s, uint8_t op, int regf, int rmf, bool mem, uint64_t ea, bool opsize);

static int x87(X64State *s, Dec *d, uint8_t op)
{
    modrm(d);
    FIN();
    return x87_body(s, op, d->regf, d->rmf, d->mem, d->ea, d->opsize);
}

// D8-DF once decoded (also run from the cache).
static int x87_body(X64State *s, uint8_t op, int regf, int rmf, bool mem, uint64_t ea, bool opsize)
{
    if (opsize && mem && ((op == 0xD9 && (regf == 4 || regf == 6)) || (op == 0xDD && (regf == 4 || regf == 6))))
        return X64_EXIT_UNSUPPORTED;   // the 16-bit environment formats
    // Arithmetic rounds as the control word says: switch the host's mode when it
    // differs from MXCSR's, which x64_run keeps it at.
    int rc = (s->fcw >> 10) & 3, sse_rc = (s->mxcsr >> 13) & 3;
    if (rc != sse_rc) host_set_rc(rc);
    int result = CONT;
    int reg = rmf;
    X87Reg v;
    int64_t i64;

    if (mem) {
        switch ((op << 3 | regf) & 0x3f) {   // (op & 7) << 3 | reg
        // D8/DC/DA/DE: ST0 = ST0 op m32fp / m64fp / m32int / m16int
        case 0x00: case 0x01: case 0x02: case 0x03: case 0x04: case 0x05: case 0x06: case 0x07:
        case 0x20: case 0x21: case 0x22: case 0x23: case 0x24: case 0x25: case 0x26: case 0x27:
        case 0x10: case 0x11: case 0x12: case 0x13: case 0x14: case 0x15: case 0x16: case 0x17:
        case 0x30: case 0x31: case 0x32: case 0x33: case 0x34: case 0x35: case 0x36: case 0x37:
            switch (op) {
            case 0xD8: v = f80_from_f32((uint32_t)mem_read(ea, 4)); break;
            case 0xDC: v = f80_from_f64(mem_read(ea, 8)); break;
            case 0xDA: v = f80_from_int((int32_t)mem_read(ea, 4)); break;
            default: v = f80_from_int((int16_t)mem_read(ea, 2)); break;
            }
            if (regf == X87_COM || regf == X87_COMP) {
                x87_compare(s, x87_get(s, 0), v, false);
                if (regf == X87_COMP) x87_pop(s);
            } else {
                x87_put(s, 0, x87_valid(s, 0) ? f80_arith(regf, x87_get(s, 0), v, s->fcw) : F80_INDEFINITE);
            }
            break;
        case 0x08: v = f80_from_f32((uint32_t)mem_read(ea, 4)); x87_push(s, f80_nan(v) ? f80_quiet(v) : v); break;   // FLD m32
        case 0x0A: case 0x0B: mem_write(ea, 4, f80_to_f32(x87_get(s, 0)));               // FST(P) m32
            if (regf == 3) x87_pop(s);
            break;
        case 0x0C: x87_load_env(s, ea); break;                                             // FLDENV
        case 0x0D: s->fcw = (uint16_t)mem_read(ea, 2); break;                              // FLDCW
        case 0x0E: x87_store_env(s, ea); s->fcw |= 0x3f; break;                            // FNSTENV
        case 0x0F: mem_write(ea, 2, s->fcw); break;                                        // FNSTCW
        case 0x18: x87_push(s, f80_from_int((int32_t)mem_read(ea, 4))); break;            // FILD m32
        case 0x19: case 0x1A: case 0x1B:                                                   // FISTTP FIST FISTP m32
            if (!f80_to_int(x87_get(s, 0), regf == 1 ? 3 : rc, &i64) || i64 != (int32_t)i64) i64 = INT32_MIN;
            mem_write(ea, 4, (uint32_t)i64);
            if (regf != 2) x87_pop(s);
            break;
        case 0x1D: memcpy(&v, (const void *)(uintptr_t)ea, 10); v.pad[0] = v.pad[1] = v.pad[2] = 0; x87_push(s, v); break;   // FLD m80
        case 0x1F: v = x87_get(s, 0); memcpy((void *)(uintptr_t)ea, &v, 10); x87_pop(s); break;                           // FSTP m80
        case 0x28: v = f80_from_f64(mem_read(ea, 8)); x87_push(s, f80_nan(v) ? f80_quiet(v) : v); break;             // FLD m64
        case 0x29: case 0x3F:                                                              // FISTTP m64 (DD /1), FISTP m64 (DF /7)
            if (!f80_to_int(x87_get(s, 0), regf == 1 ? 3 : rc, &i64)) i64 = INT64_MIN;
            mem_write(ea, 8, (uint64_t)i64);
            x87_pop(s);
            break;
        case 0x39: case 0x3A: case 0x3B:                                                   // FISTTP FIST FISTP m16
            if (!f80_to_int(x87_get(s, 0), regf == 1 ? 3 : rc, &i64) || i64 != (int16_t)i64) i64 = INT16_MIN;
            mem_write(ea, 2, (uint16_t)i64);
            if (regf != 2) x87_pop(s);
            break;
        case 0x2A: case 0x2B: mem_write(ea, 8, d2u(f80_to_double(x87_get(s, 0))));       // FST(P) m64
            if (regf == 3) x87_pop(s);
            break;
        case 0x2C:                                                                         // FRSTOR
            x87_load_env(s, ea);
            for (int i = 0; i < 8; i++) {
                memcpy(&v, (const void *)(uintptr_t)(ea + 28 + 10 * (uint64_t)i), 10);
                v.pad[0] = v.pad[1] = v.pad[2] = 0;
                *x87_st(s, i) = v;
            }
            break;
        case 0x2E:                                                                         // FNSAVE
            x87_store_env(s, ea);
            for (int i = 0; i < 8; i++) memcpy((void *)(uintptr_t)(ea + 28 + 10 * (uint64_t)i), x87_st(s, i), 10);
            x87_init(s);
            break;
        case 0x2F: mem_write(ea, 2, s->fsw); break;                                        // FNSTSW m16
        case 0x38: x87_push(s, f80_from_int((int16_t)mem_read(ea, 2))); break;            // FILD m16
        case 0x3D: x87_push(s, f80_from_int((int64_t)mem_read(ea, 8))); break;            // FILD m64
        case 0x3C: x87_push(s, x87_load_bcd(ea)); break;                                   // FBLD
        case 0x3E: x87_store_bcd(s, ea, x87_get(s, 0)); x87_pop(s); break;                // FBSTP
        default: result = X64_EXIT_UNSUPPORTED; break;
        }
    } else {
        X87Reg t0 = x87_get(s, 0), ti = x87_get(s, reg);
        switch (op) {
        case 0xD8:                                                       // ST0 = ST0 op ST(i)
            if (regf == X87_COM || regf == X87_COMP) { x87_compare(s, t0, ti, false); if (regf == X87_COMP) x87_pop(s); }
            else x87_put(s, 0, x87_valid(s, 0) && x87_valid(s, reg) ? f80_arith(regf, t0, ti, s->fcw) : F80_INDEFINITE);
            break;
        case 0xDC: case 0xDE:                                            // ST(i) = ST(i) op ST0 (SUB/SUBR and DIV/DIVR swap)
            if (op == 0xDE && regf == 3 && reg != 1) { result = X64_EXIT_ILLEGAL; break; }
            if (regf == X87_COM || regf == X87_COMP) {
                x87_compare(s, t0, ti, false);
                if (regf == X87_COMP || op == 0xDE) x87_pop(s);
                if (op == 0xDE && regf == 3) x87_pop(s);              // FCOMPP
                break;
            }
            x87_put(s, reg, x87_valid(s, 0) && x87_valid(s, reg) ? f80_arith(regf >= 4 ? regf ^ 1 : regf, ti, t0, s->fcw)
                                                                  : F80_INDEFINITE);   // an empty register: stack underflow
            if (op == 0xDE) x87_pop(s);
            break;
        case 0xD9:
            switch (regf) {
            case 0: x87_push(s, ti); break;                              // FLD ST(i)
            case 1: x87_put(s, 0, ti); x87_put(s, reg, t0); break;       // FXCH
            case 2: if (reg) result = X64_EXIT_ILLEGAL; break;           // FNOP
            case 3: x87_put(s, reg, t0); x87_pop(s); break;              // FSTP1
            default: result = x87_misc(s, (uint8_t)(0xC0 | regf << 3 | reg)); break;
            }
            break;
        case 0xDA: case 0xDB:
            if (regf < 4) {                                           // FCMOVcc
                static const uint64_t flag[4] = { FL_CF, FL_ZF, FL_CF | FL_ZF, FL_PF };
                bool c = (s->rflags & flag[regf]) != 0;
                if (op == 0xDB) c = !c;
                if (!x87_valid(s, 0) || !x87_valid(s, reg)) x87_put(s, 0, F80_INDEFINITE);
                else if (c) x87_put(s, 0, ti);
            } else if (op == 0xDA && regf == 5 && reg == 1) {         // FUCOMPP
                x87_compare(s, t0, ti, false); x87_pop(s); x87_pop(s);
            } else if (op == 0xDB && regf == 4) {
                if (reg == 2) s->fsw &= 0x7f00;                          // FNCLEX
                else if (reg == 3) x87_init(s);                          // FNINIT
                else if (reg > 4) result = X64_EXIT_ILLEGAL;             // FENI, FDISI, FSETPM: no-ops
            } else if (op == 0xDB && (regf == 5 || regf == 6)) {   // FUCOMI FCOMI
                x87_compare(s, t0, ti, true);
            } else result = X64_EXIT_ILLEGAL;
            break;
        case 0xDD:
            switch (regf) {
            case 0: s->ftw &= (uint8_t)~(1 << ((x87_top(s) + reg) & 7)); break;   // FFREE
            case 1: x87_put(s, 0, ti); x87_put(s, reg, t0); break;       // FXCH4
            case 2: case 3: x87_put(s, reg, t0); if (regf == 3) x87_pop(s); break;   // FST(P) ST(i)
            case 4: case 5: x87_compare(s, t0, ti, false); if (regf == 5) x87_pop(s); break;   // FUCOM(P)
            default: result = X64_EXIT_ILLEGAL; break;
            }
            break;
        default: // 0xDF
            switch (regf) {
            case 0: s->ftw &= (uint8_t)~(1 << ((x87_top(s) + reg) & 7)); x87_pop(s); break;   // FFREEP
            case 1: x87_put(s, 0, ti); x87_put(s, reg, t0); break;       // FXCH7
            case 2: case 3: x87_put(s, reg, t0); x87_pop(s); break;      // FSTP8/9
            case 4: if (reg) result = X64_EXIT_ILLEGAL;                   // FNSTSW AX
                    else s->gpr[X64_RAX] = (s->gpr[X64_RAX] & ~0xffffull) | s->fsw;
                    break;
            case 5: case 6: x87_compare(s, t0, ti, true); x87_pop(s); break;   // FUCOMIP FCOMIP
            default: result = X64_EXIT_ILLEGAL; break;
            }
            break;
        }
    }
    if (rc != sse_rc) host_set_rc(sse_rc);
    return result;
}

// The 0F 60-7F and D0-FF rows without a prefix: MMX.
static int mmx(X64State *s, Dec *d, uint8_t op)
{
    if (op == 0x77) { FIN(); s->ftw = 0; s->fsw &= (uint16_t)~0x3800; return CONT; }   // EMMS: all empty, TOP 0
    modrm(d);
    uint8_t imm = 0;
    if (op == 0x70 || (op >= 0x71 && op <= 0x73) || op == 0xC4 || op == 0xC5) imm = fetch8(d);
    FIN();
    return mmx_body(s, d, op, imm);
}

// The MMX instructions but EMMS, once decoded (also run from the cache).
static int mmx_body(X64State *s, Dec *d, uint8_t op, uint8_t imm)
{
    int r = d->regf;
    switch (op) {
    case 0x6C: case 0x6D: case 0xD0: case 0xD6: case 0xE6: case 0xF0: case 0xFF:
        return X64_EXIT_ILLEGAL;
    case 0x71: case 0x72: case 0x73: case 0xC5: case 0xD7: case 0xF7:
        if (d->mem) return X64_EXIT_ILLEGAL;
        break;
    case 0xE7:
        if (!d->mem) return X64_EXIT_ILLEGAL;
        break;
    }
    uint64_t src = d->mem ? 0 : mm_get(s, d->rmf);
    bool src_mm = op != 0x6E && op != 0x7E && op != 0x7F && op != 0xC4 && op != 0xE7;
    if (d->mem && src_mm) src = mem_read(d->ea, op >= 0x60 && op <= 0x62 ? 4 : 8);   // low unpacks read 32 bits
    mmx_enter(s);
    X64Vec a = {{0}}, b = {{0}};
    int i;
    switch (op) {
    case 0x6E: mm_set(s, r, get_rm(s, d, (d->rex & 8) ? 8 : 4)); return CONT;   // MOVD/MOVQ mm, r/m
    case 0x6F: mm_set(s, r, src); return CONT;                                   // MOVQ mm, mm/m64
    case 0x7E: set_rm(s, d, (d->rex & 8) ? 8 : 4, mm_get(s, r)); return CONT;   // MOVD/MOVQ r/m, mm
    case 0x7F: case 0xE7:                                                        // MOVQ mm/m64, mm; MOVNTQ
        if (d->mem) mem_write(d->ea, 8, mm_get(s, r));
        else mm_set(s, d->rmf, mm_get(s, r));
        return CONT;
    case 0x70:                                                                   // PSHUFW
        b.q[0] = src;
        for (i = 0; i < 4; i++) a.w[i] = b.w[(imm >> (2 * i)) & 3];
        mm_set(s, r, a.q[0]);
        return CONT;
    case 0x71: case 0x72: case 0x73: {
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
        default: return X64_EXIT_ILLEGAL;
        }
        a.q[0] = mm_get(s, d->rmf);
        b.q[0] = imm;
        pint(equivalent, &a, &b);
        mm_set(s, d->rmf, a.q[0]);
        return CONT;
    }
    case 0xC4: a.q[0] = mm_get(s, r); a.w[imm & 3] = (uint16_t)get_rm(s, d, 2); mm_set(s, r, a.q[0]); return CONT;   // PINSRW
    case 0xC5:                                                                   // PEXTRW
        b.q[0] = mm_get(s, d->rmf);
        set_reg(s, d, d->reg, 4, b.w[imm & 3]);
        return CONT;
    case 0xD7:                                                                   // PMOVMSKB
        b.q[0] = mm_get(s, d->rmf);
        {
            uint64_t m = 0;
            for (i = 0; i < 8; i++) m |= (uint64_t)(b.b[i] >> 7) << i;
            set_reg(s, d, d->reg, 4, m);
        }
        return CONT;
    case 0xF7: {                                                                 // MASKMOVQ
        if (d->m32 && d->addrsize) return X64_EXIT_UNSUPPORTED;   // DI
        a.q[0] = mm_get(s, r);
        b.q[0] = mm_get(s, d->rmf);
        uint64_t di = (d->addrsize || d->m32 ? s->gpr[X64_RDI] & 0xffffffffull : s->gpr[X64_RDI]) + seg_base(s, d);
        for (i = 0; i < 8; i++)
            if (b.b[i] & 0x80) mem_write(ha(s, di + (uint64_t)i), 1, a.b[i]);
        return CONT;
    }
    }
    a.q[0] = mm_get(s, r);
    b.q[0] = src;
    uint8_t e = op;
    if (op == 0x68 || op == 0x69 || op == 0x6A) { a.q[0] >>= 32; b.q[0] >>= 32; e = (uint8_t)(op - 8); }   // high unpacks
    else if (op == 0x63 || op == 0x67 || op == 0x6B) a.q[1] = b.q[0];                                      // packs
    if (!pint(e, &a, &b)) return X64_EXIT_UNSUPPORTED;
    mm_set(s, r, a.q[0]);
    return CONT;
}

// One-byte opcodes only 32-bit mode has.
static int legacy32(X64State *s, Dec *d, uint8_t op)
{
    uint64_t v, al, ah, cf, af;
    int size;
    switch (op) {
    case 0x40: case 0x41: case 0x42: case 0x43: case 0x44: case 0x45: case 0x46: case 0x47:
    case 0x48: case 0x49: case 0x4A: case 0x4B: case 0x4C: case 0x4D: case 0x4E: case 0x4F:  // INC, DEC
        FIN();
        size = osz(d);
        cf = s->rflags & FL_CF;
        v = op < 0x48 ? op_add(s, get_reg(s, d, op & 7, size), 1, 0, size) : op_sub(s, get_reg(s, d, op & 7, size), 1, 0, size);
        set_flags(s, FL_CF, cf);
        set_reg(s, d, op & 7, size, v);
        return CONT;
    case 0x06: case 0x0E: case 0x16: case 0x1E: {   // PUSH ES CS SS DS: a 16-bit store, as Intel's CPUs do
        FIN();
        size = stack_size(s, d);
        uint16_t sel = op == 0x06 ? s->seg_es : op == 0x0E ? s->seg_cs : op == 0x16 ? s->seg_ss : s->seg_ds;
        uint64_t sp = ipm(s, s->gpr[X64_RSP] - (uint64_t)size);
        mem_write(ha(s, sp), 2, sel);
        s->gpr[X64_RSP] = sp;
        return CONT;
    }
    case 0x07: case 0x17: case 0x1F:                // POP ES SS DS
        FIN();
        v = pop(s, stack_size(s, d));
        if (op == 0x07) s->seg_es = (uint16_t)v; else if (op == 0x17) s->seg_ss = (uint16_t)v; else s->seg_ds = (uint16_t)v;
        return CONT;
    case 0x27: case 0x2F: {                         // DAA, DAS
        FIN();
        uint64_t old_al = s->gpr[X64_RAX] & 0xff, old_cf = s->rflags & FL_CF;
        al = old_al;
        cf = 0;
        af = 0;
        if ((al & 15) > 9 || (s->rflags & FL_AF)) {
            cf = old_cf || (op == 0x27 ? al + 6 > 0xff : al < 6);
            al = (op == 0x27 ? al + 6 : al - 6) & 0xff;
            af = 1;
        }
        if (old_al > 0x99 || old_cf) { al = (op == 0x27 ? al + 0x60 : al - 0x60) & 0xff; cf = 1; }
        else if (op == 0x27) cf = 0;
        s->gpr[X64_RAX] = (s->gpr[X64_RAX] & ~0xffull) | al;
        set_flags(s, FL_ARITH, (cf ? FL_CF : 0) | (af ? FL_AF : 0) | szp(al, 1));
        return CONT;
    }
    case 0x37: case 0x3F: {                         // AAA, AAS
        FIN();
        uint64_t ax = s->gpr[X64_RAX] & 0xffff;
        bool adjust = (ax & 15) > 9 || (s->rflags & FL_AF);
        if (adjust) {
            ax = op == 0x37 ? ax + 0x106 : (ax - 6) & 0xffff;   // AAS: AX - 6, then AH - 1
            if (op == 0x3F) ax = (ax & 0xff) | ((((ax >> 8) - 1) & 0xff) << 8);
        }
        ax &= 0xff0f;
        s->gpr[X64_RAX] = (s->gpr[X64_RAX] & ~0xffffull) | (ax & 0xffff);
        set_flags(s, FL_CF | FL_AF, adjust ? FL_CF | FL_AF : 0);
        return CONT;
    }
    case 0x60: {                                    // PUSHA
        FIN();
        size = d->opsize ? 2 : 4;
        uint64_t sp = s->gpr[X64_RSP];
        for (int i = 0; i < 8; i++) push(s, i == X64_RSP ? sp : s->gpr[i], size);
        return CONT;
    }
    case 0x61:                                      // POPA
        FIN();
        size = d->opsize ? 2 : 4;
        for (int i = 7; i >= 0; i--) {
            v = pop(s, size);
            if (i != X64_RSP) set_reg(s, d, i, size, v);
        }
        return CONT;
    case 0xCE:                                      // INTO
        FIN();
        if (!(s->rflags & FL_OF)) return CONT;
        d->disp = 4;
        return X64_EXIT_INTERRUPT;
    case 0xD4: case 0xD5: {                         // AAM, AAD
        uint8_t imm = fetch8(d);
        FIN();
        al = s->gpr[X64_RAX] & 0xff;
        ah = (s->gpr[X64_RAX] >> 8) & 0xff;
        if (op == 0xD4) {
            if (!imm) return X64_EXIT_DIVIDE_ERROR;
            ah = al / imm;
            al %= imm;
        } else {
            al = (al + ah * imm) & 0xff;
            ah = 0;
        }
        s->gpr[X64_RAX] = (s->gpr[X64_RAX] & ~0xffffull) | (ah << 8) | al;
        set_flags(s, FL_ARITH, szp(al, 1));
        return CONT;
    }
    case 0xD6:                                      // SALC
        FIN();
        s->gpr[X64_RAX] = (s->gpr[X64_RAX] & ~0xffull) | ((s->rflags & FL_CF) ? 0xff : 0);
        return CONT;
    }
    return X64_EXIT_ILLEGAL;
}

static int step(X64State *s, const X64Env *env, Dec *d)
{
    uint8_t op;
    int map = 0, size;
    uint64_t v = 0, t;

    d->start = s->rip;
    s->insn_start = s->rip;
    d->m32 = s->mode32;
    d->gbase = s->guest_base;
    d->begin = d->p = (const uint8_t *)(uintptr_t)ha(s, s->rip);
    d->rex = 0; d->rep = 0; d->opsize = d->addrsize = d->lock = false; d->seg = -1; d->mem = false; d->addr16 = false;

    for (;;) {
        if (d->p - d->begin >= 15) return X64_EXIT_ILLEGAL;
        op = *d->p;
        if ((op & 0xF0) == 0x40 && !d->m32) { d->rex = op; d->p++; continue; }   // INC/DEC in 32-bit mode
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
    // 16-bit addressing in 32-bit mode: memory operands (modrm), and the instructions
    // that address through registers or an absolute offset. A 67 prefix on anything
    // else (a linker's "addr32 call") changes nothing.
    if (d->m32 && d->addrsize && ((op >= 0xA0 && op <= 0xAF && op != 0xA8 && op != 0xA9) || op == 0xD7 ||
                                  (op >= 0xE0 && op <= 0xE3)))
        return X64_EXIT_UNSUPPORTED;
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
        size = stack_size(s, d);
        push(s, s->gpr[(op & 7) | ((d->rex & 1) << 3)] & mask_of(size), size);
        return CONT;
    case 0x58: case 0x59: case 0x5A: case 0x5B: case 0x5C: case 0x5D: case 0x5E: case 0x5F:
        FIN();
        size = stack_size(s, d);
        v = pop(s, size);
        set_reg(s, d, (op & 7) | ((d->rex & 1) << 3), size, v);
        return CONT;
    case 0x63: // MOVSXD (ARPL in 32-bit mode)
        if (d->m32) return X64_EXIT_UNSUPPORTED;
        size = osz(d);
        modrm(d);
        FIN();
        if (size == 8) s->gpr[d->reg] = (uint64_t)(int64_t)(int32_t)get_rm(s, d, 4);
        else set_reg(s, d, d->reg, size, get_rm(s, d, size));
        return CONT;
    case 0x68: case 0x6A: // PUSH imm
        size = stack_size(s, d);
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
        if (cond(s, op & 15)) s->rip = ipm(s, d->next + t);
        return CONT;
    case 0x82: // 0x80 in 32-bit mode
        if (!d->m32) return X64_EXIT_ILLEGAL;
        /* fall through */
    case 0x80: case 0x81: case 0x83: {
        size = op == 0x80 || op == 0x82 ? 1 : osz(d);
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
        size = stack_size(s, d);
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
        size = stack_size(s, d);
        push(s, (s->rflags | 2) & mask_of(size) & ~0x30000ull, size);
        return CONT;
    case 0x9D: // POPF
        FIN();
        size = stack_size(s, d);
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
        uint64_t addr = d->addrsize || d->m32 ? fetch32(d) : fetch64(d);
        FIN();
        addr = ha(s, addr + seg_base(s, d));
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
        s->rip = pop(s, s->mode32 ? 4 : 8);
        s->gpr[X64_RSP] = ipm(s, s->gpr[X64_RSP] + release);
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
        size = stack_size(s, d);
        push(s, s->gpr[X64_RBP], size);
        set_reg(s, d, X64_RBP, size, s->gpr[X64_RSP]);
        s->gpr[X64_RSP] = ipm(s, s->gpr[X64_RSP] - frame);
        return CONT;
    }
    case 0xC9: { // LEAVE (the read first, so a fault changes nothing)
        FIN();
        size = stack_size(s, d);
        v = mem_read(ha(s, s->gpr[X64_RBP]), size);
        s->gpr[X64_RSP] = ipm(s, s->gpr[X64_RBP] + (uint64_t)size);
        set_reg(s, d, X64_RBP, size, v);
        return CONT;
    }
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
        v = (d->addrsize || d->m32 ? s->gpr[X64_RBX] & 0xffffffffull : s->gpr[X64_RBX]) + (s->gpr[X64_RAX] & 0xff);
        set_reg(s, d, X64_RAX, 1, mem_read(ha(s, v + seg_base(s, d)), 1));
        return CONT;
    case 0xD8: case 0xD9: case 0xDA: case 0xDB: case 0xDC: case 0xDD: case 0xDE: case 0xDF:
        return x87(s, d, op);
    case 0xE0: case 0xE1: case 0xE2: case 0xE3: { // LOOPNE LOOPE LOOP JRCXZ
        t = (uint64_t)(int64_t)(int8_t)fetch8(d);
        FIN();
        int asize = d->addrsize || d->m32 ? 4 : 8;
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
        if (take) s->rip = ipm(s, d->next + t);
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
        return muldiv(s, d->regf, get_rm(s, d, size), size);
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
        case 2: return call(s, env, get_rm(s, d, d->m32 ? 4 : 8), d->next);
        case 4: return jump(s, env, get_rm(s, d, d->m32 ? 4 : 8));
        case 6:
            size = stack_size(s, d);
            push(s, get_rm(s, d, size), size);
            return CONT;
        default: return X64_EXIT_UNSUPPORTED;   // far CALL and JMP
        }
    }
    case 0x8E: { // MOV Sreg, r/m16
        modrm(d);
        FIN();
        uint16_t sel = (uint16_t)get_rm(s, d, 2);
        switch (d->regf) {
        case 0: s->seg_es = sel; return CONT;
        case 2: s->seg_ss = sel; return CONT;
        case 3: s->seg_ds = sel; return CONT;
        case 4: s->seg_fs = sel; return CONT;   // the base (the TEB) stays
        case 5: s->seg_gs = sel; return CONT;
        default: return X64_EXIT_ILLEGAL;
        }
    }
    case 0x9A: case 0xEA: case 0xCA: case 0xCB: case 0xCF: // far CALL, JMP, RET; IRET
        return d->m32 || op >= 0xCA ? X64_EXIT_UNSUPPORTED : X64_EXIT_ILLEGAL;
    default:
        if (d->m32) return legacy32(s, d, op);
        return X64_EXIT_ILLEGAL;   // the opcodes 64-bit mode removed (06, 07, 27, 60, D4...)
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
        if (cond(s, op & 15)) s->rip = ipm(s, d->next + t);
        return CONT;
    case 0x90: case 0x91: case 0x92: case 0x93: case 0x94: case 0x95: case 0x96: case 0x97:
    case 0x98: case 0x99: case 0x9A: case 0x9B: case 0x9C: case 0x9D: case 0x9E: case 0x9F: // SETcc
        modrm(d);
        FIN();
        set_rm(s, d, 1, cond(s, op & 15));
        return CONT;
    case 0xA0: case 0xA8: { // PUSH FS, PUSH GS: a 16-bit store, as Intel's CPUs do
        FIN();
        size = stack_size(s, d);
        uint64_t sp = ipm(s, s->gpr[X64_RSP] - (uint64_t)size);
        mem_write(ha(s, sp), 2, op == 0xA0 ? s->seg_fs : s->seg_gs);
        s->gpr[X64_RSP] = sp;
        return CONT;
    }
    case 0xA1: case 0xA9: // POP FS, POP GS (the bases stay)
        FIN();
        v = pop(s, stack_size(s, d));
        if (op == 0xA1) s->seg_fs = (uint16_t)v; else s->seg_gs = (uint16_t)v;
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
            // __sync rather than __atomic: GCC inlines only the former for 16 bytes.
            unsigned __int128 old = __sync_val_compare_and_swap((unsigned __int128 *)(uintptr_t)d->ea, expected, desired);
            bool ok = old == expected;
            if (!ok) { s->gpr[X64_RAX] = (uint64_t)old; s->gpr[X64_RDX] = (uint64_t)(old >> 64); }
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

void x64_state_init32(X64State *s, uint64_t guest_base)
{
    x64_state_init(s);
    s->mode32 = true;
    s->guest_base = guest_base;
    s->seg_cs = 0x23;
}

void x64_fxsave(const X64State *s, void *area)
{
    fx_store(s, (uint8_t *)area);
}

void x64_fxrstor(X64State *s, const void *area)
{
    fx_load(s, (const uint8_t *)area);
}

// ---------------------------------------------------------------------------
// Lazy flags
//
// ADD, SUB, CMP, AND, OR, XOR, TEST, INC and DEC run from the cache leave their
// operands in the state instead of working out six flags each time; most are
// overwritten before anything reads them. flags_flush works them out, and
// lazy_cond answers a condition after a CMP or TEST without it.

enum { LF_NONE, LF_ADD, LF_SUB, LF_LOGIC, LF_INC, LF_DEC, LF_ADC, LF_SBB };

static inline void lazy(X64State *s, int op, int size, uint64_t a, uint64_t b, uint64_t r)
{
    s->lazy_op = (uint8_t)op;
    s->lazy_size = (uint8_t)size;
    s->lazy_a = a;
    s->lazy_b = b;
    s->lazy_r = r;
}

static void flags_flush(X64State *s)
{
    uint64_t a = s->lazy_a, b = s->lazy_b, r = s->lazy_r, sign = sign_of(s->lazy_size), f, which = FL_ARITH;
    switch (s->lazy_op) {
    case LF_NONE: return;
    case LF_ADD: case LF_INC:
        f = (r < a ? FL_CF : 0) | (((a ^ r) & (b ^ r) & sign) ? FL_OF : 0) | (((a ^ b ^ r) & 0x10) ? FL_AF : 0);
        break;
    case LF_SUB: case LF_DEC:
        f = (a < b ? FL_CF : 0) | (((a ^ b) & (a ^ r) & sign) ? FL_OF : 0) | (((a ^ b ^ r) & 0x10) ? FL_AF : 0);
        break;
    case LF_ADC:   // with lazy_carry in
        f = (r < a || (s->lazy_carry && r == a) ? FL_CF : 0) | (((a ^ r) & (b ^ r) & sign) ? FL_OF : 0)
          | (((a ^ b ^ r) & 0x10) ? FL_AF : 0);
        break;
    case LF_SBB:
        f = (a < b || (s->lazy_carry && a == b) ? FL_CF : 0) | (((a ^ b) & (a ^ r) & sign) ? FL_OF : 0)
          | (((a ^ b ^ r) & 0x10) ? FL_AF : 0);
        break;
    default:   // LF_LOGIC: CF, OF and AF clear
        f = 0;
        break;
    }
    f |= szp(r, s->lazy_size);
    if (s->lazy_op == LF_INC || s->lazy_op == LF_DEC) which &= ~FL_CF;   // CF stays
    s->rflags = (s->rflags & ~which) | (f & which);
    s->lazy_op = LF_NONE;
}

void x64_flush_flags(X64State *s)
{
    flags_flush(s);
}

// CF alone, for INC and DEC, which keep it.
static inline uint64_t lazy_cf(const X64State *s)
{
    switch (s->lazy_op) {
    case LF_ADD: return s->lazy_r < s->lazy_a ? FL_CF : 0;
    case LF_SUB: return s->lazy_a < s->lazy_b ? FL_CF : 0;
    case LF_ADC: return s->lazy_r < s->lazy_a || (s->lazy_carry && s->lazy_r == s->lazy_a) ? FL_CF : 0;
    case LF_SBB: return s->lazy_a < s->lazy_b || (s->lazy_carry && s->lazy_a == s->lazy_b) ? FL_CF : 0;
    case LF_LOGIC: return 0;
    default: return s->rflags & FL_CF;
    }
}

static inline bool lazy_cond(X64State *s, int cc)
{
    uint64_t a = s->lazy_a, b = s->lazy_b, r = s->lazy_r, sign = sign_of(s->lazy_size);
    bool v;
    if (s->lazy_op == LF_SUB && (cc >> 1) != 5) {
        int size = s->lazy_size;
        switch (cc >> 1) {
        case 0: v = ((a ^ b) & (a ^ r) & sign) != 0; break;   // O
        case 1: v = a < b; break;                             // B
        case 2: v = r == 0; break;                            // Z
        case 3: v = a <= b; break;                            // BE
        case 4: v = (r & sign) != 0; break;                   // S
        case 6: v = sext(a, size) < sext(b, size); break;     // L
        default: v = sext(a, size) <= sext(b, size); break;   // LE
        }
        return (cc & 1) ? !v : v;
    }
    if (s->lazy_op == LF_LOGIC && (cc >> 1) != 5) {
        switch (cc >> 1) {
        case 0: case 1: v = false; break;                     // O B
        case 2: case 3: v = r == 0; break;                    // Z BE
        case 4: case 6: v = (r & sign) != 0; break;           // S L
        default: v = r == 0 || (r & sign) != 0; break;        // LE
        }
        return (cc & 1) ? !v : v;
    }
    flags_flush(s);
    return cond(s, cc);
}

// ---------------------------------------------------------------------------
// The decoded-instruction cache (X64Env.cache)
//
// One 64-byte entry per instruction, direct-mapped by address. An entry for one
// of the common forms below holds it decoded; any other instruction's entry says
// so, and step() decodes and runs it as before. Before an entry runs, the
// instruction's bytes are compared with the ones it was decoded from.

enum {
    FX_EMPTY, FX_SLOW,
    FX_ALU_RM_R, FX_ALU_R_RM, FX_ALU_RM_I,   // aux: the ALU operation
    FX_TEST_RM_R, FX_TEST_RM_I,
    FX_MOV_RM_R, FX_MOV_R_RM, FX_MOV_RM_I, FX_LEA,
    FX_INC, FX_DEC,
    FX_PUSH_R, FX_POP_R, FX_PUSH_I, FX_PUSH_RM,
    FX_JCC, FX_JMP, FX_CALL, FX_RET,         // imm: the target (JCC JMP CALL), the bytes to release (RET)
    FX_JMP_RM, FX_CALL_RM,
    FX_MOVZX, FX_MOVSX,                      // aux: the source size
    FX_SETCC, FX_CMOV,                       // aux: the condition
    FX_IMUL_R_RM, FX_IMUL_R_RM_I,
    FX_SHIFT,                                // aux: the operation; imm: the count, or 256 for CL
    FX_SHIFTD,                               // SHLD (aux 1) SHRD (aux 0); imm as FX_SHIFT
    FX_GROUP3,                               // aux: NOT NEG MUL IMUL DIV IDIV (2 to 7)
    FX_NOP, FX_CWDE, FX_CDQ,
    // Decoded here, run by the x87, SSE and MMX code step() uses (aux: the opcode; size:
    // the prefixes, 1 for 66, 2 for F3, 4 for F2; imm: the 8-bit immediate).
    FX_X87, FX_SSE,
};

#define FX_NONE 0xff     // no base or index register
#define FX_RIPREL 0xfe   // base: RIP

typedef struct FastEntry {
    uint64_t tag;          // guest address, bit 63 set for 32-bit mode
    uint64_t w0, w1;       // the instruction's bytes, and which of them count
    uint64_t m0, m1;
    uint64_t imm;
    int32_t disp;
    uint8_t len, kind, size, aux;
    uint8_t reg, rm;       // ModRM.reg; the register operand (when not memory)
    uint8_t base, index, scale, seg, rex;
    uint8_t mem;
} FastEntry;

_Static_assert(sizeof(FastEntry) <= 64, "a cache entry is 64 bytes");

static inline uint64_t fx_tag(const X64State *s, uint64_t rip)
{
    return s->mode32 ? rip | 0x8000000000000000ull : rip;
}

static inline FastEntry *fx_entry(void *cache, uint64_t rip)
{
    return (FastEntry *)cache + ((rip ^ (rip >> 13)) & (X64_CACHE_ENTRIES - 1));
}

static inline uint64_t load64(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return v; }

// Whether the bytes at p are still the entry's.
static inline bool fx_same(const FastEntry *e, const uint8_t *p)
{
    if (((uintptr_t)p & 4095) <= 4096 - 16) {  // both words on one page: no fault past its end
        if (!e->m1) return !((load64(p) ^ e->w0) & e->m0);
        return !(((load64(p) ^ e->w0) & e->m0) | ((load64(p + 8) ^ e->w1) & e->m1));
    }
    uint8_t bytes[16];
    memcpy(bytes, &e->w0, 8);
    memcpy(bytes + 8, &e->w1, 8);
    return !memcmp(bytes, p, e->len);
}

static inline uint32_t rd16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static inline uint32_t rd32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }

// ModRM, SIB and displacement into the entry; returns what follows them.
static const uint8_t *fx_modrm(FastEntry *e, const uint8_t *p, bool m32, uint8_t rex, int *regf)
{
    uint8_t m = *p++;
    int mod = m >> 6, rmf = m & 7;
    *regf = (m >> 3) & 7;
    e->reg = (uint8_t)(*regf | ((rex & 4) << 1));
    e->base = e->index = FX_NONE;
    e->scale = 0;
    e->disp = 0;
    if (mod == 3) {
        e->mem = 0;
        e->rm = (uint8_t)(rmf | ((rex & 1) << 3));
        return p;
    }
    e->mem = 1;
    if (rmf == 4) {
        uint8_t sib = *p++;
        int index = ((sib >> 3) & 7) | ((rex & 2) << 2);
        if (index != 4) { e->index = (uint8_t)index; e->scale = (uint8_t)(sib >> 6); }
        if ((sib & 7) == 5 && mod == 0) { e->disp = (int32_t)rd32(p); p += 4; }
        else e->base = (uint8_t)((sib & 7) | ((rex & 1) << 3));
    } else if (rmf == 5 && mod == 0) {
        if (!m32) e->base = FX_RIPREL;
        e->disp = (int32_t)rd32(p);
        p += 4;
    } else {
        e->base = (uint8_t)(rmf | ((rex & 1) << 3));
    }
    if (mod == 1) e->disp = (int8_t)*p++;
    else if (mod == 2) { e->disp = (int32_t)rd32(p); p += 4; }
    return p;
}

// An immediate of the operand size: 8 or 16 bits as they are, 32 sign-extended.
static inline const uint8_t *fx_imm(FastEntry *e, const uint8_t *p, int size)
{
    if (size == 1) { e->imm = *p; return p + 1; }
    if (size == 2) { e->imm = rd16(p); return p + 2; }
    e->imm = (uint64_t)(int64_t)(int32_t)rd32(p);
    return p + 4;
}

// Decodes the instruction at p (guest address rip) into e: one of the forms
// below, or FX_SLOW. Either way the entry keeps the bytes the decision was made
// from (for FX_SLOW, the ones read before it), and is decoded again when they
// change.
static void fx_decode(const X64State *s, FastEntry *e, uint64_t rip, const uint8_t *p0)
{
    const uint8_t *p = p0;
    bool m32 = s->mode32, opsize = false;
    uint8_t rex = 0, op, rep = 0;
    int seg = 0, regf = 0, size = 4;
    e->tag = fx_tag(s, rip);
    e->kind = FX_SLOW;
    e->mem = 0;
    e->base = e->index = FX_NONE;
    e->disp = 0;
    e->imm = 0;
    int kind = FX_SLOW;
    for (;;) {
        if (p - p0 >= 8) goto done;                                // a long run of prefixes: step()
        op = *p;
        if (!m32 && (op & 0xF0) == 0x40) { rex = op; p++; continue; }
        if (op == 0x66) opsize = true;
        else if (op == 0xF2 || op == 0xF3) rep = op;                    // the last one counts (SSE only, below)
        else if (op == 0x64 || op == 0x65) seg = op == 0x64 ? 4 : 5;
        else if (op != 0x26 && op != 0x2E && op != 0x36 && op != 0x3E) break;   // F0 67: step()
        rex = 0;   // a REX prefix only counts right before the opcode
        p++;
    }
    p++;
    size = (rex & 8) ? 8 : opsize ? 2 : 4;
    int stack = opsize ? 2 : m32 ? 4 : 8;
    if (op < 0x40 && (op & 7) < 6 && op != 0x0F) {             // ALU
        e->aux = (uint8_t)(op >> 3);
        int form = op & 7;
        if (!(form & 1)) size = 1;
        if (form >= 4) { p = fx_imm(e, p, size); e->rm = X64_RAX; kind = FX_ALU_RM_I; }
        else { p = fx_modrm(e, p, m32, rex, &regf); kind = form < 2 ? FX_ALU_RM_R : FX_ALU_R_RM; }
    } else if (op >= 0x40 && op <= 0x4F) {                     // INC, DEC (32-bit mode)
        e->rm = op & 7;
        kind = op < 0x48 ? FX_INC : FX_DEC;
    } else if (op >= 0x50 && op <= 0x57) { e->reg = (uint8_t)((op & 7) | ((rex & 1) << 3)); size = stack; kind = FX_PUSH_R; }
    else if (op >= 0x58 && op <= 0x5F) { e->reg = (uint8_t)((op & 7) | ((rex & 1) << 3)); size = stack; kind = FX_POP_R; }
    else if (op == 0x68 || op == 0x6A) {
        size = stack;
        if (op == 0x6A) e->imm = (uint64_t)(int64_t)(int8_t)*p++;
        else p = fx_imm(e, p, size);
        kind = FX_PUSH_I;
    } else if (op == 0x69 || op == 0x6B) {
        p = fx_modrm(e, p, m32, rex, &regf);
        if (op == 0x6B) e->imm = (uint64_t)(int64_t)(int8_t)*p++;
        else if (size == 2) { e->imm = (uint64_t)(int64_t)(int16_t)rd16(p); p += 2; }
        else { e->imm = (uint64_t)(int64_t)(int32_t)rd32(p); p += 4; }
        kind = FX_IMUL_R_RM_I;
    } else if (op >= 0x70 && op <= 0x7F) {
        int8_t rel = (int8_t)*p++;
        e->aux = op & 15;
        e->imm = (uint64_t)(int64_t)rel;
        kind = opsize ? FX_SLOW : FX_JCC;
    } else if (op == 0x80 || op == 0x81 || op == 0x83) {
        if (op == 0x80) size = 1;
        p = fx_modrm(e, p, m32, rex, &regf);
        if (op == 0x83) e->imm = (uint64_t)(int64_t)(int8_t)*p++;
        else p = fx_imm(e, p, size);
        e->aux = (uint8_t)regf;
        kind = FX_ALU_RM_I;
    } else if (op == 0x84 || op == 0x85) {
        if (op == 0x84) size = 1;
        p = fx_modrm(e, p, m32, rex, &regf);
        kind = FX_TEST_RM_R;
    } else if (op >= 0x88 && op <= 0x8B) {
        if (!(op & 1)) size = 1;
        p = fx_modrm(e, p, m32, rex, &regf);
        kind = op < 0x8A ? FX_MOV_RM_R : FX_MOV_R_RM;
    } else if (op == 0x8D) {
        p = fx_modrm(e, p, m32, rex, &regf);
        if (e->mem) kind = FX_LEA;
    } else if (op == 0x90) {
        if (!(rex & 1)) kind = FX_NOP;                         // with REX.B: XCHG R8, RAX
    } else if (op == 0x98) kind = FX_CWDE;
    else if (op == 0x99) kind = FX_CDQ;
    else if (op == 0xA8 || op == 0xA9) {
        if (op == 0xA8) size = 1;
        p = fx_imm(e, p, size);
        e->rm = X64_RAX;
        kind = FX_TEST_RM_I;
    } else if (op >= 0xB0 && op <= 0xBF) {
        e->rm = (uint8_t)((op & 7) | ((rex & 1) << 3));
        if (op < 0xB8) { size = 1; e->imm = *p++; }
        else if (size == 8) { e->imm = load64(p); p += 8; }
        else if (size == 2) { e->imm = rd16(p); p += 2; }
        else { e->imm = rd32(p); p += 4; }
        kind = FX_MOV_RM_I;
    } else if (op == 0xC0 || op == 0xC1 || op == 0xD0 || op == 0xD1 || op == 0xD2 || op == 0xD3) {
        if (!(op & 1)) size = 1;
        p = fx_modrm(e, p, m32, rex, &regf);
        e->imm = op <= 0xC1 ? *p++ : op <= 0xD1 ? 1 : 256;
        e->aux = (uint8_t)regf;
        kind = FX_SHIFT;
    } else if (op == 0xC2 || op == 0xC3) {
        if (op == 0xC2) { e->imm = rd16(p); p += 2; }
        kind = opsize ? FX_SLOW : FX_RET;
    } else if (op == 0xC6 || op == 0xC7) {
        if (op == 0xC6) size = 1;
        p = fx_modrm(e, p, m32, rex, &regf);
        if (regf == 0) { p = fx_imm(e, p, size); kind = FX_MOV_RM_I; }
    } else if (op == 0xE8 || op == 0xE9 || op == 0xEB) {
        int64_t rel;
        if (op == 0xEB) rel = (int8_t)*p++;
        else { rel = (int32_t)rd32(p); p += 4; }
        e->imm = (uint64_t)rel;
        kind = opsize ? FX_SLOW : op == 0xE8 ? FX_CALL : FX_JMP;
    } else if (op == 0xF6 || op == 0xF7) {
        if (op == 0xF6) size = 1;
        p = fx_modrm(e, p, m32, rex, &regf);
        if (regf < 2) { p = fx_imm(e, p, size); kind = FX_TEST_RM_I; }
        else { e->aux = (uint8_t)regf; kind = FX_GROUP3; }
    } else if (op >= 0xD8 && op <= 0xDF) {
        p = fx_modrm(e, p, m32, rex, &regf);
        e->aux = op;
        kind = FX_X87;
    } else if (op == 0xFE || op == 0xFF) {
        if (op == 0xFE) size = 1;
        p = fx_modrm(e, p, m32, rex, &regf);
        if (regf == 0) kind = FX_INC;
        else if (regf == 1) kind = FX_DEC;
        else if (op == 0xFF && regf == 2) kind = FX_CALL_RM;
        else if (op == 0xFF && regf == 4) kind = FX_JMP_RM;
        else if (op == 0xFF && regf == 6) { size = stack; kind = FX_PUSH_RM; }
    } else if (op == 0x0F) {
        op = *p++;
        if (op == 0x0D || (op >= 0x18 && op <= 0x1F)) { p = fx_modrm(e, p, m32, rex, &regf); kind = FX_NOP; }   // hint NOPs
        else if (op >= 0x40 && op <= 0x4F) { p = fx_modrm(e, p, m32, rex, &regf); e->aux = op & 15; kind = FX_CMOV; }
        else if (op >= 0x80 && op <= 0x8F) {
            e->imm = (uint64_t)(int64_t)(int32_t)rd32(p);
            p += 4;
            e->aux = op & 15;
            kind = opsize ? FX_SLOW : FX_JCC;
        } else if (op >= 0x90 && op <= 0x9F) { p = fx_modrm(e, p, m32, rex, &regf); e->aux = op & 15; size = 1; kind = FX_SETCC; }
        else if (op == 0xAF) { p = fx_modrm(e, p, m32, rex, &regf); kind = FX_IMUL_R_RM; }
        else if ((op >= 0x10 && op <= 0x17) || (op >= 0x28 && op <= 0x2F) || (op >= 0x50 && op <= 0x5F) ||
                 (op >= 0xC2 && op <= 0xC6 && op != 0xC3) || (op >= 0x60 && op <= 0x7F && op != 0x77) || op >= 0xD0) {
            // What step() hands to sse() (and, without a prefix, MMX); EMMS stays with it.
            p = fx_modrm(e, p, m32, rex, &regf);
            if (op == 0x70 || op == 0x71 || op == 0x72 || op == 0x73 || op == 0xC2 || op == 0xC4 || op == 0xC5 || op == 0xC6)
                e->imm = *p++;
            e->aux = op;
            kind = FX_SSE;
        }
        else if (op == 0xA4 || op == 0xA5 || op == 0xAC || op == 0xAD) {
            p = fx_modrm(e, p, m32, rex, &regf);
            e->imm = (op & 1) ? 256 : *p++;
            e->aux = op < 0xA8;
            kind = FX_SHIFTD;
        }
        else if (op == 0xB6 || op == 0xB7 || op == 0xBE || op == 0xBF) {
            p = fx_modrm(e, p, m32, rex, &regf);
            e->aux = (op & 1) ? 2 : 1;
            kind = op >= 0xBE ? FX_MOVSX : FX_MOVZX;
        }
    }
done:;
    size_t len = (size_t)(p - p0);
    if (len > 15) { kind = FX_SLOW; len = 15; }   // step() reports it; 15 bytes decided that
    if (rep && kind != FX_SSE) kind = FX_SLOW;     // REP strings, PAUSE, POPCNT...
    if (kind == FX_X87 || kind == FX_SSE) size = (opsize ? 1 : 0) | (rep == 0xF3 ? 2 : rep == 0xF2 ? 4 : 0);
    if (kind == FX_JCC || kind == FX_JMP || kind == FX_CALL)   // the target, as EIP or RIP holds it
        e->imm = ipm(s, rip + len + e->imm);
    e->len = (uint8_t)len;
    e->size = (uint8_t)size;
    e->seg = (uint8_t)seg;
    e->rex = rex;
    uint8_t bytes[16] = { 0 };
    memcpy(bytes, p0, len);
    memcpy(&e->w0, bytes, 8);
    memcpy(&e->w1, bytes + 8, 8);
    e->m0 = len >= 8 ? ~0ull : (1ull << (8 * len)) - 1;
    e->m1 = len <= 8 ? 0 : (1ull << (8 * (len - 8))) - 1;
    e->kind = (uint8_t)kind;
}

static inline uint64_t fx_getr(const X64State *s, const FastEntry *e, int reg, int size)
{
    if (size == 1) {
        if (!e->rex && reg >= 4 && reg < 8) return (s->gpr[reg - 4] >> 8) & 0xff;
        return s->gpr[reg] & 0xff;
    }
    return s->gpr[reg] & mask_of(size);
}

static inline void fx_setr(X64State *s, const FastEntry *e, int reg, int size, uint64_t v)
{
    switch (size) {
    case 1:
        if (!e->rex && reg >= 4 && reg < 8) s->gpr[reg - 4] = (s->gpr[reg - 4] & ~0xff00ull) | ((v & 0xff) << 8);
        else s->gpr[reg] = (s->gpr[reg] & ~0xffull) | (v & 0xff);
        break;
    case 2: s->gpr[reg] = (s->gpr[reg] & ~0xffffull) | (v & 0xffff); break;
    case 4: s->gpr[reg] = v & 0xffffffffull; break;
    default: s->gpr[reg] = v; break;
    }
}

// The memory operand's offset (LEA's result), and its host address.
static inline uint64_t fx_offset(const X64State *s, const FastEntry *e, uint64_t next)
{
    uint64_t off = (uint64_t)(int64_t)e->disp;
    if (e->base == FX_RIPREL) off += next;
    else if (e->base != FX_NONE) off += s->gpr[e->base];
    if (e->index != FX_NONE) off += s->gpr[e->index] << e->scale;
    return s->mode32 ? (uint32_t)off : off;
}

static inline uint64_t fx_ea(const X64State *s, const FastEntry *e, uint64_t next)
{
    uint64_t a = fx_offset(s, e, next) + (e->seg == 4 ? s->fs_base : e->seg == 5 ? s->gs_base : 0);
    return s->mode32 ? s->guest_base + (uint32_t)a : a;
}

static inline uint64_t fx_get(const X64State *s, const FastEntry *e, uint64_t ea, int size)
{
    return e->mem ? mem_read(ea, size) : fx_getr(s, e, e->rm, size);
}

static inline void fx_set(X64State *s, const FastEntry *e, uint64_t ea, int size, uint64_t v)
{
    if (e->mem) mem_write(ea, size, v);
    else fx_setr(s, e, e->rm, size, v);
}

// An ALU operation, its flags left lazy (ADC and SBB work them out).
static inline uint64_t fx_alu(X64State *s, int op, uint64_t a, uint64_t b, int size)
{
    uint64_t m = mask_of(size), r;
    a &= m;
    b &= m;
    switch (op) {
    case ALU_ADD: r = (a + b) & m; lazy(s, LF_ADD, size, a, b, r); return r;
    case ALU_SUB: case ALU_CMP: r = (a - b) & m; lazy(s, LF_SUB, size, a, b, r); return r;
    case ALU_AND: r = a & b; break;
    case ALU_OR: r = a | b; break;
    case ALU_XOR: r = a ^ b; break;
    default: {   // ADC SBB: CF in
        uint64_t c = lazy_cf(s) ? 1 : 0;
        r = (op == ALU_ADC ? a + b + c : a - b - c) & m;
        lazy(s, op == ALU_ADC ? LF_ADC : LF_SBB, size, a, b, r);
        s->lazy_carry = (uint8_t)c;
        return r;
    }
    }
    lazy(s, LF_LOGIC, size, 0, 0, r);
    return r;
}

static inline void fx_imul_flags(X64State *s, uint64_t r, __int128 full, int size)
{
    bool over = (__int128)sext(r, size) != full;
    s->lazy_op = LF_NONE;   // every arithmetic flag is set here
    set_flags(s, FL_ARITH, (over ? FL_CF | FL_OF : 0) | szp(r, size));
}

#define FX_FALLBACK (-2)   // run this one with step() instead; nothing has changed

// Runs a decoded instruction: what step() does for it.
static int fx_run(X64State *s, const X64Env *env, const FastEntry *e)
{
    uint64_t next = ipm(s, s->rip + e->len), ea = 0, v, t;
    int size = e->size;
    if (e->mem) ea = fx_ea(s, e, next);
    switch (e->kind) {
    case FX_ALU_RM_R:
        if (e->aux == ALU_CMP) fx_alu(s, ALU_CMP, fx_get(s, e, ea, size), fx_getr(s, e, e->reg, size), size);
        else fx_set(s, e, ea, size, fx_alu(s, e->aux, fx_get(s, e, ea, size), fx_getr(s, e, e->reg, size), size));
        break;
    case FX_ALU_R_RM:
        v = fx_alu(s, e->aux, fx_getr(s, e, e->reg, size), fx_get(s, e, ea, size), size);
        if (e->aux != ALU_CMP) fx_setr(s, e, e->reg, size, v);
        break;
    case FX_ALU_RM_I:
        if (e->aux == ALU_CMP) fx_alu(s, ALU_CMP, fx_get(s, e, ea, size), e->imm, size);
        else fx_set(s, e, ea, size, fx_alu(s, e->aux, fx_get(s, e, ea, size), e->imm, size));
        break;
    case FX_TEST_RM_R: fx_alu(s, ALU_AND, fx_get(s, e, ea, size), fx_getr(s, e, e->reg, size), size); break;
    case FX_TEST_RM_I: fx_alu(s, ALU_AND, fx_get(s, e, ea, size), e->imm, size); break;
    case FX_MOV_RM_R: fx_set(s, e, ea, size, fx_getr(s, e, e->reg, size)); break;
    case FX_MOV_R_RM: fx_setr(s, e, e->reg, size, fx_get(s, e, ea, size)); break;
    case FX_MOV_RM_I: fx_set(s, e, ea, size, e->imm); break;
    case FX_LEA: fx_setr(s, e, e->reg, size, fx_offset(s, e, next)); break;
    case FX_INC: case FX_DEC: {
        uint64_t m = mask_of(size), a = fx_get(s, e, ea, size) & m;
        uint64_t r = (e->kind == FX_INC ? a + 1 : a - 1) & m;
        s->rflags = (s->rflags & ~FL_CF) | lazy_cf(s);
        lazy(s, e->kind == FX_INC ? LF_INC : LF_DEC, size, a, 1, r);
        fx_set(s, e, ea, size, r);
        break;
    }
    case FX_PUSH_R: push(s, s->gpr[e->reg] & mask_of(size), size); break;
    case FX_POP_R: v = pop(s, size); fx_setr(s, e, e->reg, size, v); break;
    case FX_PUSH_I: push(s, e->imm & mask_of(size), size); break;
    case FX_PUSH_RM: push(s, fx_get(s, e, ea, size), size); break;
    case FX_JCC:
        s->rip = (s->lazy_op ? lazy_cond(s, e->aux) : cond(s, e->aux)) ? e->imm : next;
        return CONT;
    case FX_JMP:
        if (native(env, e->imm)) return FX_FALLBACK;
        s->rip = e->imm;
        return CONT;
    case FX_CALL:
        if (native(env, e->imm)) return FX_FALLBACK;
        push(s, next, s->mode32 ? 4 : 8);
        s->rip = e->imm;
        return CONT;
    case FX_JMP_RM: case FX_CALL_RM:
        t = ipm(s, fx_get(s, e, ea, s->mode32 ? 4 : 8));
        if (native(env, t)) return FX_FALLBACK;
        if (e->kind == FX_CALL_RM) push(s, next, s->mode32 ? 4 : 8);
        s->rip = t;
        return CONT;
    case FX_RET:
        s->rip = pop(s, s->mode32 ? 4 : 8);
        s->gpr[X64_RSP] = ipm(s, s->gpr[X64_RSP] + e->imm);
        return CONT;
    case FX_MOVZX: case FX_MOVSX:
        v = fx_get(s, e, ea, e->aux);
        fx_setr(s, e, e->reg, size, e->kind == FX_MOVSX ? (uint64_t)sext(v, e->aux) : v);
        break;
    case FX_SETCC: fx_set(s, e, ea, 1, s->lazy_op ? lazy_cond(s, e->aux) : cond(s, e->aux)); break;
    case FX_CMOV:
        v = fx_get(s, e, ea, size);
        fx_setr(s, e, e->reg, size, (s->lazy_op ? lazy_cond(s, e->aux) : cond(s, e->aux)) ? v : fx_getr(s, e, e->reg, size));
        break;
    case FX_IMUL_R_RM: {
        __int128 full = (__int128)sext(fx_getr(s, e, e->reg, size), size) * sext(fx_get(s, e, ea, size), size);
        uint64_t r = (uint64_t)full & mask_of(size);
        fx_imul_flags(s, r, full, size);
        fx_setr(s, e, e->reg, size, r);
        break;
    }
    case FX_IMUL_R_RM_I: {
        __int128 full = (__int128)sext(fx_get(s, e, ea, size), size) * (int64_t)e->imm;
        uint64_t r = (uint64_t)full & mask_of(size);
        fx_imul_flags(s, r, full, size);
        fx_setr(s, e, e->reg, size, r);
        break;
    }
    case FX_SHIFT: {
        unsigned count = e->imm == 256 ? (unsigned)(s->gpr[X64_RCX] & 0xff) : (unsigned)e->imm;
        unsigned masked = count & (size == 8 ? 63 : 31);
        flags_flush(s);
        v = shift(s, e->aux, fx_get(s, e, ea, size), count, size);
        if (masked || size == 4) fx_set(s, e, ea, size, v);
        break;
    }
    case FX_SHIFTD: {
        unsigned count = e->imm == 256 ? (unsigned)(s->gpr[X64_RCX] & 0xff) : (unsigned)e->imm;
        unsigned masked = count & (size == 8 ? 63 : 31);
        flags_flush(s);
        v = double_shift(s, e->aux, fx_get(s, e, ea, size), fx_getr(s, e, e->reg, size), count, size);
        if (masked || size == 4) fx_set(s, e, ea, size, v);
        break;
    }
    case FX_GROUP3: {
        uint64_t a = fx_get(s, e, ea, size);
        if (e->aux == 2) fx_set(s, e, ea, size, ~a & mask_of(size));                 // NOT
        else if (e->aux == 3) fx_set(s, e, ea, size, fx_alu(s, ALU_SUB, 0, a, size));  // NEG: CF is a != 0
        else if (muldiv(s, e->aux, a, size) != CONT) return FX_FALLBACK;               // a divide error: step() raises it
        break;
    }
    case FX_X87:
        if (s->lazy_op) flags_flush(s);   // FCOMI sets EFLAGS, FCMOVcc reads them
        s->rip = next;
        return x87_body(s, e->aux, e->reg & 7, e->rm & 7, e->mem != 0, ea, (e->size & 1) != 0);
    case FX_SSE: {
        // The decoder's state, as step() would leave it for these.
        Dec d;
        d.begin = d.p = NULL;   // nothing more is fetched
        d.addrsize = d.lock = d.riprel = d.addr16 = false;
        d.base = d.index = -1;
        d.scale = 0;
        d.disp = 0;
        d.off = d.ea = 0;
        d.rm = d.rmf = 0;
        d.start = s->rip;
        d.next = next;
        d.m32 = s->mode32;
        d.gbase = s->guest_base;
        d.rex = e->rex;
        d.rep = (e->size & 2) ? 0xF3 : (e->size & 4) ? 0xF2 : 0;
        d.opsize = (e->size & 1) != 0;
        d.seg = e->seg ? e->seg : -1;
        d.regf = e->reg & 7;
        d.reg = e->reg;
        d.mem = e->mem != 0;
        if (d.mem) {
            d.mod = 0;
            d.base = e->base == FX_NONE || e->base == FX_RIPREL ? -1 : e->base;
            d.riprel = e->base == FX_RIPREL;
            d.index = e->index == FX_NONE ? -1 : e->index;
            d.scale = e->scale;
            d.disp = e->disp;
            d.off = fx_offset(s, e, next);
            d.ea = ea;
        } else {
            d.mod = 3;
            d.rm = e->rm;
            d.rmf = e->rm & 7;
        }
        if (s->lazy_op) flags_flush(s);   // COMISS and the like set EFLAGS
        s->rip = next;
        int pfx = d.rep == 0xF3 ? 2 : d.rep == 0xF2 ? 3 : d.opsize ? 1 : 0;
        uint8_t op = e->aux;
        if (pfx == 0 && ((op >= 0x60 && op <= 0x7F) || op >= 0xD0 || op == 0xC4 || op == 0xC5)) return mmx_body(s, &d, op, (uint8_t)e->imm);
        return sse_body(s, &d, op, pfx, (uint8_t)e->imm);
    }
    case FX_NOP: break;
    case FX_CWDE: fx_setr(s, e, X64_RAX, size, (uint64_t)sext(s->gpr[X64_RAX], size / 2)); break;
    case FX_CDQ: fx_setr(s, e, X64_RDX, size, (s->gpr[X64_RAX] & sign_of(size)) ? ~0ull : 0); break;
    }
    s->rip = next;
    return CONT;
}

X64Exit x64_run(X64State *s, const X64Env *env, uint64_t budget, X64ExitInfo *info)
{
    int saved_rounding = host_get_rc();
    host_set_rc((int)((s->mxcsr >> 13) & 3));
    Dec d;
    int r;
    size_t length;
    void *cache = env->cache;
    for (;;) {
        if (s->rip == env->stop_rip || budget == 0) {
            r = s->rip == env->stop_rip ? X64_EXIT_STOP : X64_EXIT_BUDGET;
            if (info) { memset(info, 0, sizeof *info); info->rip = s->rip; }
            break;
        }
        if (budget != UINT64_MAX) budget--;
        uint64_t start = s->rip;
        if (cache) {
            FastEntry *e = fx_entry(cache, start);
            const uint8_t *code = (const uint8_t *)(uintptr_t)ha(s, start);
            s->insn_start = start;
            if (e->tag != fx_tag(s, start) || e->kind == FX_EMPTY || !fx_same(e, code))
                fx_decode(s, e, start, code);   // new, or changed, code
            if (e->kind != FX_SLOW) {
                r = fx_run(s, env, e);
                if (r == CONT) { s->instructions++; continue; }
                if (r != FX_FALLBACK) { length = e->len; goto stopped; }   // an exit, as step() gives it
            }
        }
        flags_flush(s);   // step() reads and writes rflags itself
        r = step(s, env, &d);
        if (r == CONT) { s->instructions++; continue; }
        length = (size_t)(d.p - d.begin);
    stopped:
        if (r == X64_EXIT_DIVIDE_ERROR || r == X64_EXIT_ILLEGAL || r == X64_EXIT_PRIVILEGED ||
            r == X64_EXIT_UNSUPPORTED)
            s->rip = start;
        else
            s->instructions++;
        if (info) {
            if (length > sizeof info->bytes) length = sizeof info->bytes;
            memset(info, 0, sizeof *info);
            info->rip = start;
            info->length = (uint8_t)length;
            memcpy(info->bytes, (const void *)(uintptr_t)ha(s, start), length);
            if (r == X64_EXIT_INTERRUPT) info->vector = (uint8_t)d.disp;
        }
        break;
    }
    flags_flush(s);
    host_set_rc(saved_rounding);
    return (X64Exit)r;
}

void x64_return_from_native(X64State *s, uint64_t rax)
{
    if (s->mode32) {
        s->gpr[X64_RAX] = (uint32_t)rax;
        s->gpr[X64_RDX] = rax >> 32;
        s->rip = pop(s, 4);
    } else {
        s->gpr[X64_RAX] = rax;
        s->rip = pop(s, 8);
    }
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
        out[3] = (1u << 0) | (1u << 4) | (1u << 8) | (1u << 15) | (1u << 19) | (1u << 23)
               | (1u << 24) | (1u << 25) | (1u << 26);  // FPU TSC CX8 CMOV CLFSH MMX FXSR SSE SSE2
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
    s->lazy_op = 0;
    memcpy(&s->mxcsr, c + CTX_MXCSR, 4);
    memcpy(&s->seg_cs, c + CTX_SEGCS, 2);
    memcpy(&s->seg_ds, c + CTX_SEGDS, 2);
    memcpy(&s->seg_es, c + CTX_SEGES, 2);
    memcpy(&s->seg_fs, c + CTX_SEGFS, 2);
    memcpy(&s->seg_gs, c + CTX_SEGGS, 2);
    memcpy(&s->seg_ss, c + CTX_SEGSS, 2);
    uint32_t mxcsr = s->mxcsr;
    fx_load(s, c + CTX_FLTSAVE);
    s->mxcsr = mxcsr;   // CONTEXT.MxCsr is the one that counts
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
    fx_store(s, c + CTX_FLTSAVE);
}
