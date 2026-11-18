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

#define CODE    0x30000000ul
#define ANCHOR  0x30100000ul
#define MEM     0x30200000ul
#define MEMSZ   0x4000ul
#define STACK   0x30300000ul
#define STACKSZ 0x2000ul

enum {
    RCX_SMALL = 1 << 0, R8_SMALL = 1 << 1, BITOFF = 1 << 2, POPF = 1 << 3, XLAT = 1 << 4,
    APPROX = 1 << 5, NOMEM = 1 << 6, SSE = 1 << 7, RCX17 = 1 << 8, DIV = 1 << 9,
};

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

static void dump_input(void)
{
    for (int i = 0; i < 16; i++)
        fprintf(stderr, "    %s=%016llx%s", reg_names[i], (unsigned long long)IN->gpr[i], i % 4 == 3 ? "\n" : "");
    fprintf(stderr, "    flags=%llx mxcsr=%x\n", (unsigned long long)IN->flags, IN->mxcsr);
    for (int i = 0; i < 4; i++)
        fprintf(stderr, "    xmm%d=%016llx:%016llx\n", i, (unsigned long long)IN->xmm[i].q[1], (unsigned long long)IN->xmm[i].q[0]);
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
    X64Env env = { .stop_rip = CODE + pre + len };
    X64ExitInfo info;
    X64Exit exit = X64_EXIT_BUDGET;
    fault_signal = 0;
    if (sigsetjmp(fault_jump, 1) == 0) exit = x64_run(&s, &env, 100000, &info);

    int k = 0;
    why[0] = 0;
    if (fault_signal) { snprintf(why, size, "interpreter faulted with signal %d", fault_signal); return; }
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

#if defined(__x86_64__)
extern const struct Template fuzz_templates[];
extern const uint64_t fuzz_template_count;
extern const uint8_t stub_pre_start[], stub_pre_end[], stub_post_start[], stub_post_end[];
static uint8_t mem_native[MEMSZ], stack_native[STACKSZ];

// Runs the snippet natively from IN; OUT and the memory regions hold the results.
static int run_native(void)
{
    unsigned host_mxcsr = _mm_getcsr();
    fault_signal = 0;
    if (sigsetjmp(fault_jump, 1) == 0) ((void (*)(void))CODE)();
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

static int replay(FILE *f)
{
    char magic[8];
    uint64_t header[4];
    if (fread(magic, 8, 1, f) != 1 || memcmp(magic, "X64FUZZ1", 8) || fread(header, sizeof header, 1, f) != 1) {
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
