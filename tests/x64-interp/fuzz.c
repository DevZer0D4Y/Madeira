// SPDX-License-Identifier: GPL-3.0-or-later
// Madeira Converter Exception: see LICENSE-EXCEPTION.md
//
// Differential test of X64Interpreter against a real x86-64 CPU.
//
// Each snippet from gen-fuzz-insns.py runs natively at a fixed address with
// random registers, flags, XMM registers, MXCSR and memory, then through the
// interpreter at the same address from the same state. The registers, the
// defined flags, MXCSR's control bits and both memory regions must match.
//
//   fuzz run TRIALS SEED [FILTER]    x86-64 Linux: native and interpreted, compared
//   fuzz record FILE TRIALS SEED     x86-64 Linux: native results, written to FILE
//   fuzz replay FILE                 any 64-bit Linux: interpreted, compared with FILE
//
// record and replay take the comparison to an ARM64 host (the iPhone's
// architecture, here under QEMU): the inputs are regenerated from the seed, so
// only the native results travel.

#define _GNU_SOURCE
#include "X64Interpreter.h"

#include <math.h>
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#if defined(__x86_64__)
#include <asm/prctl.h>
#include <sys/syscall.h>
#include <xmmintrin.h>
#endif

#include "fuzz-common.h"

// Random input state and memory for one trial. Uses the generator in the same
// order on every host, which is what lets replay regenerate a recorded input.
static void setup_input(uint64_t flags)
{
    for (int i = 0; i < 16; i++) IN->gpr[i] = rnd_int();
    for (int i = 0; i < 16; i++) rnd_vec(&IN->xmm[i]);
    IN->flags = 0x202 | (rnd() & 0xcd5);
    IN->mxcsr = 0x1f80;
    if (flags & SSE) {
        IN->mxcsr |= (uint32_t)(rnd() % 4) << 13;
        if (rnd() % 4 == 0) IN->mxcsr |= 0x8000;
        if (rnd() % 4 == 0) IN->mxcsr |= 0x40;
    }
    IN->gpr[X64_RSI] = MEM + 0x1000;
    IN->gpr[X64_RDI] = MEM + 0x2000;
    IN->gpr[X64_RSP] = STACK + 0x1000;
    if (flags & RCX_SMALL) IN->gpr[X64_RCX] = rnd() % 41;
    if (flags & RCX17) IN->gpr[X64_RCX] = rnd() % 17;
    if (flags & R8_SMALL) IN->gpr[X64_R8] = rnd() % 16;
    if (flags & BITOFF) IN->gpr[X64_RBX] = (uint64_t)((int64_t)(rnd() % 401) - 200);
    if (flags & XLAT) IN->gpr[X64_RBX] = MEM + 0x1800;
    if (flags & POPF) IN->gpr[X64_RBX] = (rnd() & 0x200ed5) | 2;
    for (size_t i = 0; i < MEMSZ; i += 8) { uint64_t v = rnd_int(); memcpy((uint8_t *)MEM + i, &v, 8); }
    for (size_t i = 0; i < STACKSZ; i += 8) { uint64_t v = rnd(); memcpy((uint8_t *)STACK + i, &v, 8); }
    if (flags & DIV) {
        int size = (int)((flags >> 16) & 15);
        bool is_signed = (flags >> 20) & 1, in_mem = (flags >> 21) & 1;
        uint64_t m = size == 8 ? ~0ull : (1ull << (size * 8)) - 1;
        uint64_t divisor = rnd_int() & m;
        if (!divisor) divisor = 7;
        if (is_signed) {
            // High half = the sign of the low half: the quotient fits but for MIN / -1.
            uint64_t low = IN->gpr[X64_RAX] & m;
            int64_t sl = size == 8 ? (int64_t)low : size == 4 ? (int32_t)low : size == 2 ? (int16_t)low : (int8_t)low;
            int64_t sd = size == 8 ? (int64_t)divisor : size == 4 ? (int32_t)divisor : size == 2 ? (int16_t)divisor : (int8_t)divisor;
            if (sd == -1 && (uint64_t)sl == ((1ull << (size * 8 - 1)) | ~m)) divisor = 3;
            uint64_t high = sl < 0 ? m : 0;
            if (size == 1) IN->gpr[X64_RAX] = (IN->gpr[X64_RAX] & ~0xffffull) | (high << 8) | low;
            else IN->gpr[X64_RDX] = (IN->gpr[X64_RDX] & ~m) | high;
        } else {
            uint64_t high = rnd_int() % divisor;
            if (size == 1) IN->gpr[X64_RAX] = (IN->gpr[X64_RAX] & ~0xff00ull) | (high << 8);
            else IN->gpr[X64_RDX] = (IN->gpr[X64_RDX] & ~m) | high;
        }
        if (in_mem) memcpy((uint8_t *)MEM + 0x1000, &divisor, (size_t)size);
        else IN->gpr[X64_RBX] = (IN->gpr[X64_RBX] & ~m) | divisor;
    }
}

static const char *const reg_names[16] = { "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
                                           "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15" };

// Every other run goes through the decoded-instruction cache, which keeps its
// entries from run to run: the same snippet again hits them, the next snippet at
// the same address replaces them. The rest run without it (step() alone).
static _Alignas(64) uint8_t fuzz_cache[X64_CACHE_SIZE];
static void *next_cache(void)
{
    static unsigned runs;
    return (runs++ & 1) ? NULL : fuzz_cache;
}

// Every third run goes in pieces of 1 to 5 instructions, each ending in X64_EXIT_BUDGET after
// exactly that many, or sooner in another exit (a block can stop and resume part of the way
// through): the result must be the same. Returns the piece size for the next run, or 0 for one.
static volatile int piece_miscount;
static unsigned next_piece(void)
{
    static unsigned runs;
    runs++;
    return runs % 3 == 2 ? 1 + runs % 5 : 0;
}

static X64Exit run_in_pieces(X64State *s, const X64Env *env, X64ExitInfo *info, unsigned piece)
{
    uint64_t left = 100000;
    piece_miscount = 0;
    if (!piece) return x64_run(s, env, left, info);
    for (;;) {
        uint64_t before = s->instructions, n = piece < left ? piece : left;
        X64Exit exit = x64_run(s, env, n, info);
        uint64_t ran = s->instructions - before;
        if (exit == X64_EXIT_BUDGET ? ran != n : ran > n) piece_miscount = 1;   // never more than the budget
        if (exit != X64_EXIT_BUDGET) return exit;
        if (!(left -= n)) return exit;
    }
}

static void dump_input(void)
{
    for (int i = 0; i < 16; i++)
        fprintf(stderr, "    %s=%016llx%s", reg_names[i], (unsigned long long)IN->gpr[i], i % 4 == 3 ? "\n" : "");
    fprintf(stderr, "    flags=%llx mxcsr=%x\n", (unsigned long long)IN->flags, IN->mxcsr);
    for (int i = 0; i < 4; i++)
        fprintf(stderr, "    xmm%d=%016llx:%016llx\n", i, (unsigned long long)IN->xmm[i].q[1], (unsigned long long)IN->xmm[i].q[0]);
}




static uint64_t fs_base, gs_base = MEM;
static uint8_t mem_snapshot[MEMSZ], stack_snapshot[STACKSZ];

// Runs the snippet at CODE + pre through the interpreter from IN, after
// restoring the trial's memory. Returns the failure, or "" when it matches
// `expect` (registers) and the expected memory: the bytes, or their hashes.
static void interpret_and_compare(size_t pre, size_t len, uint64_t undef, uint64_t flags, const struct NState *expect,
                                  const uint8_t *mem_expect, const uint8_t *stack_expect,
                                  uint64_t mem_hash, uint64_t stack_hash, char *why, size_t size)
{
    memcpy((void *)MEM, mem_snapshot, MEMSZ);
    memcpy((void *)STACK, stack_snapshot, STACKSZ);
    X64State s;
    x64_state_init(&s);
    memcpy(s.gpr, IN->gpr, sizeof s.gpr);
    memcpy(s.xmm, IN->xmm, sizeof s.xmm);
    s.rflags = IN->flags;
    s.mxcsr = IN->mxcsr;
    s.fcw = 0x37f;   // Linux's x87 default, which the native run has
    s.fs_base = fs_base;
    s.gs_base = gs_base;
    s.rip = CODE + pre;
    X64Env env = { .stop_rip = CODE + pre + len, .cache = next_cache() };
    X64ExitInfo info;
    X64Exit exit = X64_EXIT_BUDGET;
    volatile unsigned piece = next_piece();
    fault_signal = 0;
    if (sigsetjmp(fault_jump, 1) == 0) exit = run_in_pieces(&s, &env, &info, piece);

    int k = 0;
    why[0] = 0;
    if (fault_signal) { snprintf(why, size, "interpreter faulted with signal %d", fault_signal); return; }
    if (piece_miscount) { snprintf(why, size, "runs of %u instructions ran another number", piece); return; }
    if (exit != X64_EXIT_STOP) {
        k = snprintf(why, size, " exit %s at %llx:", x64_exit_name(exit), (unsigned long long)info.rip);
        for (int i = 0; i < info.length && k < (int)size - 4; i++) k += snprintf(why + k, size - (size_t)k, " %02x", info.bytes[i]);
        return;
    }
    for (int i = 0; i < 16; i++)
        if (s.gpr[i] != expect->gpr[i] && k < (int)size - 80)
            k += snprintf(why + k, size - (size_t)k, " %s: native %llx interp %llx;", reg_names[i],
                          (unsigned long long)expect->gpr[i], (unsigned long long)s.gpr[i]);
    uint64_t fmask = 0xcd5ull & ~undef;
    if (((s.rflags ^ expect->flags) & fmask) && k < (int)size - 80)
        k += snprintf(why + k, size - (size_t)k, " flags: native %llx interp %llx (compared %llx);",
                      (unsigned long long)(expect->flags & fmask), (unsigned long long)(s.rflags & fmask), (unsigned long long)fmask);
    for (int i = 0; i < 16; i++) {
        bool same = !memcmp(&s.xmm[i], &expect->xmm[i], 16);
        if (!same && (flags & APPROX)) {
            same = true;
            for (int l = 0; l < 4; l++) same = same && approx_equal(s.xmm[i].d[l], expect->xmm[i].d[l]);
        }
        if (!same && k < (int)size - 80)
            k += snprintf(why + k, size - (size_t)k, " xmm%d: native %016llx:%016llx interp %016llx:%016llx;", i,
                          (unsigned long long)expect->xmm[i].q[1], (unsigned long long)expect->xmm[i].q[0],
                          (unsigned long long)s.xmm[i].q[1], (unsigned long long)s.xmm[i].q[0]);
    }
    if (((s.mxcsr ^ expect->mxcsr) & 0xffc0) && k < (int)size - 80)
        k += snprintf(why + k, size - (size_t)k, " mxcsr: native %x interp %x;", expect->mxcsr, s.mxcsr);
    if (flags & NOMEM) return;
    if (mem_expect) {
        for (size_t i = 0; i < MEMSZ; i++)
            if (((uint8_t *)MEM)[i] != mem_expect[i] && k < (int)size - 80) {
                k += snprintf(why + k, size - (size_t)k, " mem+%zx: native %02x interp %02x;", i, mem_expect[i], ((uint8_t *)MEM)[i]);
                break;
            }
        for (size_t i = 0; i < STACKSZ; i++)
            if (((uint8_t *)STACK)[i] != stack_expect[i] && k < (int)size - 80) {
                k += snprintf(why + k, size - (size_t)k, " stack+%zx: native %02x interp %02x;", i, stack_expect[i], ((uint8_t *)STACK)[i]);
                break;
            }
    } else {
        if (hash((void *)MEM, MEMSZ) != mem_hash && k < (int)size - 80) k += snprintf(why + k, size - (size_t)k, " memory differs;");
        if (hash((void *)STACK, STACKSZ) != stack_hash && k < (int)size - 80) k += snprintf(why + k, size - (size_t)k, " stack differs;");
    }
}



#if defined(__x86_64__)
extern const struct Template fuzz_templates[];
extern const uint64_t fuzz_template_count;
extern const uint8_t stub_pre_start[], stub_pre_end[], stub_post_start[], stub_post_end[];
static uint8_t mem_native[MEMSZ], stack_native[STACKSZ];

// Runs the snippet natively from IN; OUT and the memory regions hold the results.
static int run_native(void)
{
    unsigned host_mxcsr = _mm_getcsr();
    void (*volatile snippet)(void) = (void (*)(void))CODE;   // not a direct call: see fuzz32.c
    fault_signal = 0;
    if (sigsetjmp(fault_jump, 1) == 0) snippet();
    _mm_setcsr(host_mxcsr);
    __asm__ volatile("fninit");
    return fault_signal;
}

static int run_or_record(long trials, const char *filter, FILE *record)
{
    size_t pre = (size_t)(stub_pre_end - stub_pre_start), post = (size_t)(stub_post_end - stub_post_start);
    long failures = 0, failed_templates = 0, runs = 0, native_faults = 0;
    if (record) {
        fwrite("X64FUZZ1", 1, 8, record);
        uint64_t header[4] = { fuzz_template_count, (uint64_t)trials, rng, pre };
        fwrite(header, sizeof header, 1, record);
    }
    for (uint64_t n = 0; n < fuzz_template_count; n++) {
        const struct Template *t = &fuzz_templates[n];
        if (filter && !strstr(t->name, filter)) continue;
        size_t len = (size_t)(t->end - t->start);
        uint8_t *code = (uint8_t *)CODE;
        memcpy(code, stub_pre_start, pre);
        memcpy(code + pre, t->start, len);
        memcpy(code + pre + len, stub_post_start, post);
        __builtin___clear_cache((char *)code, (char *)code + pre + len + post);
        if (record) {
            uint32_t name_len = (uint32_t)strlen(t->name), code_len = (uint32_t)len;
            fwrite(&name_len, 4, 1, record); fwrite(t->name, 1, name_len, record);
            fwrite(&code_len, 4, 1, record); fwrite(t->start, 1, len, record);
            fwrite(&t->undef, 8, 1, record); fwrite(&t->flags, 8, 1, record);
        }
        int template_failures = 0;
        for (long trial = 0; trial < trials && (record || template_failures < 3); trial++) {
            setup_input(t->flags);
            memcpy(mem_snapshot, (void *)MEM, MEMSZ);
            memcpy(stack_snapshot, (void *)STACK, STACKSZ);
            int fault = run_native();
            if (record) {
                uint64_t hashes[2] = { hash((void *)MEM, MEMSZ), hash((void *)STACK, STACKSZ) };
                uint8_t faulted = (uint8_t)fault;
                fwrite(OUT, sizeof *OUT, 1, record);
                fwrite(hashes, sizeof hashes, 1, record);
                fwrite(&faulted, 1, 1, record);
                runs++;
                native_faults += fault != 0;
                continue;
            }
            if (fault) {
                native_faults++;
                fprintf(stderr, "NATIVE FAULT %d in [%s]\n", fault, t->name);
                dump_input();
                template_failures++;
                continue;
            }
            memcpy(mem_native, (void *)MEM, MEMSZ);
            memcpy(stack_native, (void *)STACK, STACKSZ);
            char why[640];
            interpret_and_compare(pre, len, t->undef, t->flags, OUT, mem_native, stack_native, 0, 0, why, sizeof why);
            runs++;
            if (why[0]) {
                failures++;
                if (!template_failures) failed_templates++;
                template_failures++;
                fprintf(stderr, "FAIL [%s]%s\n", t->name, why);
                dump_input();
            }
        }
    }
    printf("%llu snippets, %ld %s, %ld failures in %ld snippets, %ld native faults\n",
           (unsigned long long)fuzz_template_count, runs, record ? "recorded" : "runs", failures, failed_templates, native_faults);
    return failures || native_faults ? 1 : 0;
}
#endif


// x87 registers equal, or (APPROX) equal as doubles to about 9 digits; a
// transcendental's native result outside double's exponent range is not
// compared: the interpreter computes them in double (docs/NO-JIT.md).
static bool f80_close(const uint8_t *a, const uint8_t *b, bool approx)
{
    if (!memcmp(a, b, 10)) return true;
    if (!approx) return false;
    uint64_t ma, mb;
    uint16_t ea, eb;
    memcpy(&ma, a, 8); memcpy(&ea, a + 8, 2);
    memcpy(&mb, b, 8); memcpy(&eb, b + 8, 2);
    int native_exponent = (eb & 0x7fff) - 16383;
    if (mb && (eb & 0x7fff) != 0x7fff && (native_exponent > 1023 || native_exponent < -1022)) return true;
    long double x = ldexpl((long double)ma, (ea & 0x7fff) - 16383 - 63), y = ldexpl((long double)mb, (eb & 0x7fff) - 16383 - 63);
    if (ea & 0x8000) x = -x;
    if (eb & 0x8000) y = -y;
    if (isnan((double)x) && isnan((double)y)) return true;
    return fabsl(x - y) <= fabsl(y) * 1e-9L;
}

// x64_run with faults caught; nothing of the caller's lives across the jump.
static X64Exit run_guarded(X64State *s, const X64Env *env, X64ExitInfo *info)
{
    volatile X64Exit exit = X64_EXIT_BUDGET;
    volatile unsigned piece = next_piece();
    fault_signal = 0;
    if (sigsetjmp(fault_jump, 1) == 0) exit = run_in_pieces(s, env, info, piece);
    return exit;
}

static int replay32(FILE *f)
{
    uint64_t header[4];
    uint16_t sel[6];
    if (fread(header, sizeof header, 1, f) != 1 || fread(sel, sizeof sel, 1, f) != 1) return 2;
    uint64_t count = header[0], trials = header[1], pre = header[3];
    rng = header[2];
    long failures = 0, failed_templates = 0, runs = 0, skipped = 0;
    static const char *const names32[8] = { "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi" };
    for (uint64_t n = 0; n < count; n++) {
        uint32_t name_len, code_len;
        char name[512];
        uint64_t undef, flags;
        if (fread(&name_len, 4, 1, f) != 1 || name_len >= sizeof name || fread(name, 1, name_len, f) != name_len) return 2;
        name[name_len] = 0;
        if (fread(&code_len, 4, 1, f) != 1 || code_len > 4096 || fread((uint8_t *)CODE + pre, 1, code_len, f) != code_len) return 2;
        if (fread(&undef, 8, 1, f) != 1 || fread(&flags, 8, 1, f) != 1) return 2;
        int template_failures = 0;
        for (uint64_t trial = 0; trial < trials; trial++) {
            struct NState32 expect;
            uint64_t hashes[2];
            uint8_t faulted;
            if (fread(&expect, sizeof expect, 1, f) != 1 || fread(hashes, sizeof hashes, 1, f) != 1 || fread(&faulted, 1, 1, f) != 1)
                return 2;
            setup_input32(flags);
            if (faulted || template_failures >= 3) { skipped++; continue; }
            X64State s;
            x64_state_init32(&s, 0);
            for (int i = 0; i < 8; i++) s.gpr[i] = IN32->gpr[i];
            s.rflags = IN32->eflags;
            x64_fxrstor(&s, IN32->fx);
            s.seg_cs = sel[0]; s.seg_ds = sel[1]; s.seg_es = sel[2]; s.seg_ss = sel[3]; s.seg_fs = sel[4]; s.seg_gs = sel[5];
            s.rip = CODE + pre;
            X64Env env = { .stop_rip = CODE + pre + code_len, .cache = next_cache() };
            X64ExitInfo info;
            X64Exit exit = run_guarded(&s, &env, &info);
            runs++;

            char why[1024];
            int k = 0;
            why[0] = 0;
            if (fault_signal) snprintf(why, sizeof why, " interpreter faulted with signal %d", fault_signal);
            else if (piece_miscount) snprintf(why, sizeof why, " runs of a few instructions ran another number");
            else if (exit != X64_EXIT_STOP) {
                k = snprintf(why, sizeof why, " exit %s at %llx:", x64_exit_name(exit), (unsigned long long)info.rip);
                for (int i = 0; i < info.length; i++) k += snprintf(why + k, sizeof why - (size_t)k, " %02x", info.bytes[i]);
            } else {
                uint8_t fx[512];
                memset(fx, 0, sizeof fx);
                x64_fxsave(&s, fx);
                for (int i = 0; i < 8; i++) {
                    uint32_t mask = 0xffffffffu;
                    if ((flags & FSWAX) && (i == 0 || i == 3)) mask = 0xffff7d00u | ((flags & CMPC1) ? 0x200 : 0);
                    if (((uint32_t)s.gpr[i] ^ expect.gpr[i]) & mask)
                        k += snprintf(why + k, sizeof why - (size_t)k, " %s: native %x interp %x;", names32[i], expect.gpr[i], (uint32_t)s.gpr[i]);
                }
                uint32_t fmask = 0xcd5u & ~(uint32_t)undef;
                if (((uint32_t)s.rflags ^ expect.eflags) & fmask)
                    k += snprintf(why + k, sizeof why - (size_t)k, " eflags: native %x interp %x;", expect.eflags & fmask, (uint32_t)s.rflags & fmask);
                uint16_t ncw, icw, nsw, isw;
                memcpy(&ncw, expect.fx, 2); memcpy(&icw, fx, 2);
                memcpy(&nsw, expect.fx + 2, 2); memcpy(&isw, fx + 2, 2);
                // C0-C3 are undefined after the transcendental instructions (APPROX): TOP and C2 only.
                uint16_t swmask = (flags & APPROX) ? 0x3c00 : 0x7d00 | ((flags & CMPC1) ? 0x200 : 0);
                if (ncw != icw) k += snprintf(why + k, sizeof why - (size_t)k, " fcw: native %x interp %x;", ncw, icw);
                if ((nsw ^ isw) & swmask) k += snprintf(why + k, sizeof why - (size_t)k, " fsw: native %x interp %x;", nsw, isw);
                if (expect.fx[4] != fx[4]) k += snprintf(why + k, sizeof why - (size_t)k, " ftw: native %x interp %x;", expect.fx[4], fx[4]);
                int top = (nsw >> 11) & 7;
                for (int i = 0; i < 8 && k < 800; i++) {
                    bool used = (expect.fx[4] >> ((top + i) & 7)) & 1;
                    const uint8_t *a = fx + 32 + 16 * i, *b = expect.fx + 32 + 16 * i;
                    if ((used || (flags & MMX)) && !f80_close(a, b, (flags & APPROX) != 0)) {
                        uint64_t ma, mb; uint16_t ea, eb;
                        memcpy(&ma, a, 8); memcpy(&ea, a + 8, 2); memcpy(&mb, b, 8); memcpy(&eb, b + 8, 2);
                        k += snprintf(why + k, sizeof why - (size_t)k, " st%d: native %04x:%016llx interp %04x:%016llx;", i,
                                      eb, (unsigned long long)mb, ea, (unsigned long long)ma);
                    }
                }
                for (int i = 0; i < 8 && k < 800; i++)
                    if (memcmp(fx + 160 + 16 * i, expect.fx + 160 + 16 * i, 16))
                        k += snprintf(why + k, sizeof why - (size_t)k, " xmm%d differs;", i);
                uint32_t nmx, imx;
                memcpy(&nmx, expect.fx + 24, 4); memcpy(&imx, fx + 24, 4);
                if ((nmx ^ imx) & 0xffc0) k += snprintf(why + k, sizeof why - (size_t)k, " mxcsr: native %x interp %x;", nmx, imx);
                if (!(flags & NOMEM)) {
                    if (hash((void *)MEM, MEMSZ) != hashes[0]) k += snprintf(why + k, sizeof why - (size_t)k, " memory differs;");
                    if (hash((void *)STACK, STACKSZ) != hashes[1]) k += snprintf(why + k, sizeof why - (size_t)k, " stack differs;");
                }
            }
            if (why[0]) {
                failures++;
                if (!template_failures) failed_templates++;
                template_failures++;
                if (template_failures == 1) {
                    fprintf(stderr, "FAIL [%s]%s\n", name, why);
                    if (flags & (X87 | MMX)) {   // the x87 input: control word and the stack from ST0
                        uint16_t cw, sw;
                        memcpy(&cw, IN32->fx, 2); memcpy(&sw, IN32->fx + 2, 2);
                        fprintf(stderr, "  input fcw %04x tags %02x:", cw, IN32->fx[4]);
                        for (int i = 0; i < 4; i++) {
                            const uint8_t *r = IN32->fx + 32 + 16 * i;
                            uint64_t m; uint16_t e;
                            memcpy(&m, r, 8); memcpy(&e, r + 8, 2);
                            fprintf(stderr, " st%d %04x:%016llx", i, e, (unsigned long long)m);
                        }
                        uint32_t m32; uint64_t m64; uint32_t i32;
                        memcpy(&m32, (void *)(MEM + 0x1010), 4); memcpy(&m64, (void *)(MEM + 0x1010), 8); memcpy(&i32, (void *)(MEM + 0x1018), 4);
                        fprintf(stderr, " [esi+16] %08x / %016llx [esi+24] %08x\n", m32, (unsigned long long)m64, i32);
                    }
                }
            }
        }
    }
    printf("replayed %llu 32-bit snippets: %ld runs, %ld failures in %ld snippets, %ld skipped\n",
           (unsigned long long)count, runs, failures, failed_templates, skipped);
    return failures ? 1 : 0;
}

static int replay(FILE *f)
{
    char magic[8];
    uint64_t header[4];
    if (fread(magic, 8, 1, f) != 1) return 2;
    if (!memcmp(magic, "X86FUZZ1", 8)) return replay32(f);
    if (memcmp(magic, "X64FUZZ1", 8) || fread(header, sizeof header, 1, f) != 1) {
        fprintf(stderr, "not a recording\n");
        return 2;
    }
    uint64_t count = header[0], trials = header[1], pre = header[3];
    rng = header[2];
    long failures = 0, failed_templates = 0, runs = 0, skipped = 0;
    for (uint64_t n = 0; n < count; n++) {
        uint32_t name_len, code_len;
        char name[512];
        uint64_t undef, flags;
        if (fread(&name_len, 4, 1, f) != 1 || name_len >= sizeof name || fread(name, 1, name_len, f) != name_len) return 2;
        name[name_len] = 0;
        if (fread(&code_len, 4, 1, f) != 1 || code_len > 4096 || fread((uint8_t *)CODE + pre, 1, code_len, f) != code_len) return 2;
        if (fread(&undef, 8, 1, f) != 1 || fread(&flags, 8, 1, f) != 1) return 2;
        bool skip = strstr(name, "fs:") != NULL;   // reads this process's TLS, which differs per host
        int template_failures = 0;
        for (uint64_t trial = 0; trial < trials; trial++) {
            struct NState expect;
            uint64_t hashes[2];
            uint8_t faulted;
            if (fread(&expect, sizeof expect, 1, f) != 1 || fread(hashes, sizeof hashes, 1, f) != 1 || fread(&faulted, 1, 1, f) != 1)
                return 2;
            setup_input(flags);
            if (skip || faulted || template_failures >= 3) { skipped++; continue; }
            memcpy(mem_snapshot, (void *)MEM, MEMSZ);
            memcpy(stack_snapshot, (void *)STACK, STACKSZ);
            char why[640];
            interpret_and_compare((size_t)pre, code_len, undef, flags, &expect, NULL, NULL, hashes[0], hashes[1], why, sizeof why);
            runs++;
            if (why[0]) {
                failures++;
                if (!template_failures) failed_templates++;
                template_failures++;
                fprintf(stderr, "FAIL [%s]%s\n", name, why);
                dump_input();
            }
        }
    }
    printf("replayed %llu snippets: %ld runs, %ld failures in %ld snippets, %ld skipped\n",
           (unsigned long long)count, runs, failures, failed_templates, skipped);
    return failures ? 1 : 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: fuzz run TRIALS SEED [FILTER] | record FILE TRIALS SEED | replay FILE\n");
        return 2;
    }
    bool native = strcmp(argv[1], "replay") != 0;
    if (!map_fixed(CODE, 0x10000, PROT_READ | PROT_WRITE | (native ? PROT_EXEC : 0)) ||
        !map_fixed(ANCHOR, 0x1000, PROT_READ | PROT_WRITE) || !map_fixed(MEM, MEMSZ, PROT_READ | PROT_WRITE) ||
        !map_fixed(STACK, STACKSZ, PROT_READ | PROT_WRITE)) {
        perror("mmap");
        return 2;
    }
    catch_faults();
    if (!native) {
        FILE *f = argc > 2 ? fopen(argv[2], "rb") : NULL;
        if (!f) { perror("replay"); return 2; }
        int r = replay(f);
        fclose(f);
        return r;
    }
#if defined(__x86_64__)
    syscall(SYS_arch_prctl, ARCH_GET_FS, &fs_base);
    // GS points at MEM, as it points at the TEB on Windows (Linux leaves it unused).
    syscall(SYS_arch_prctl, ARCH_SET_GS, MEM);
    syscall(SYS_arch_prctl, ARCH_GET_GS, &gs_base);
    if (!strcmp(argv[1], "run") && argc >= 4) {
        rng = strtoull(argv[3], NULL, 0);
        if (!rng) rng = 1;
        return run_or_record(atol(argv[2]), argc > 4 ? argv[4] : NULL, NULL);
    }
    if (!strcmp(argv[1], "record") && argc >= 5) {
        FILE *f = fopen(argv[2], "wb");
        if (!f) { perror(argv[2]); return 2; }
        rng = strtoull(argv[4], NULL, 0);
        if (!rng) rng = 1;
        int r = run_or_record(atol(argv[3]), NULL, f);
        fclose(f);
        return r;
    }
#endif
    fprintf(stderr, "run and record need an x86-64 host\n");
    return 2;
}
