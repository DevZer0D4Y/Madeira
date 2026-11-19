// SPDX-License-Identifier: GPL-3.0-or-later
// Madeira Converter Exception: see LICENSE-EXCEPTION.md
//
// Runs code blobs from build/signed-arm64/rewrite.py's check (tests/host/check-arm64-teb-rewrite.py)
// on ARM64 Linux (qemu-aarch64): each case is one instruction that uses x18, as the
// original and as rewritten (a branch to a trampoline that reads the TEB from a fake
// TSD through TPIDR_EL0). Both runs start from the same registers, flags and memory;
// the registers, flags and memory after them must match, except x18 itself, which the
// rewritten code must leave alone.
//
// Input (stdin): u32 count; the prologue and epilogue (0x200 bytes, for 0x100 in the
// blob); then per case: u32 original[2], u32 rewritten[2] (the instruction or the branch
// to its trampoline, then a branch to the epilogue), u32 n, u32 trampoline[n], 32 u64
// (x0-x30 and NZCV), u64 the TEB's offset in the TSD, u64 a seed for the memory.
// The blob: the case at 0, the prologue at 0x100, the epilogue at 0x200 and its two
// literals at 0x300, the trampolines at 0x1000, the 4-byte slot with the offset at 0x2000.
// Output: one line per case, "ok" or what differed.
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define BLOB_SIZE 0x3000
#define TEB_SIZE 0x18000
#define SCRATCH_SIZE 0x10000

void run_case(uint64_t *in, void *code);
__asm__(
    ".text\n"
    ".globl run_case\n"
    "run_case:\n"                                   // x0 = 33 u64s (x0-x30, nzcv, tpidr_el0), x1 = the blob
    "    sub sp, sp, #112\n"
    "    stp x19, x20, [sp, #0]\n"
    "    stp x21, x22, [sp, #16]\n"
    "    stp x23, x24, [sp, #32]\n"
    "    stp x25, x26, [sp, #48]\n"
    "    stp x27, x28, [sp, #64]\n"
    "    stp x29, x30, [sp, #80]\n"
    "    mrs x2, tpidr_el0\n"
    "    stp x2, x18, [sp, #96]\n"
    "    ldr x2, [x0, #256]\n"
    "    msr tpidr_el0, x2\n"
    "    add x1, x1, #0x100\n"
    "    br x1\n"                                    // the prologue loads every register
    ".globl run_case_done\n"
    "run_case_done:\n"                              // the epilogue stored them, and jumps here
    "    ldp x2, x18, [sp, #96]\n"
    "    msr tpidr_el0, x2\n"
    "    ldp x19, x20, [sp, #0]\n"
    "    ldp x21, x22, [sp, #16]\n"
    "    ldp x23, x24, [sp, #32]\n"
    "    ldp x25, x26, [sp, #48]\n"
    "    ldp x27, x28, [sp, #64]\n"
    "    ldp x29, x30, [sp, #80]\n"
    "    add sp, sp, #112\n"
    "    ret\n");
extern char run_case_done[];

static uint64_t out_regs[33];

static void read_exact(void *p, size_t n)
{
    if (fread(p, 1, n, stdin) != n) {
        fprintf(stderr, "short input\n");
        exit(2);
    }
}

static uint8_t common[0x200];

static void fill(uint8_t *p, size_t n, uint64_t seed)
{
    uint64_t x = seed | 1;
    for (size_t i = 0; i < n; i += 8) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        memcpy(p + i, &x, 8);
    }
}

// Runs one case: returns the registers in out_regs and the memory in teb and scratch.
static void run(uint8_t *exec, const uint32_t *words, const uint32_t *tramp, uint32_t n, const uint64_t *regs,
                uint64_t tsd_offset, uint64_t seed, uint64_t *tsd, uint8_t *teb, uint8_t *scratch)
{
    uint64_t in[33];
    memset(exec, 0, BLOB_SIZE);
    memcpy(exec, words, 8);
    memcpy(exec + 0x100, common, sizeof(common));
    uint64_t lits[2] = { (uint64_t)(uintptr_t)out_regs, (uint64_t)(uintptr_t)run_case_done };
    memcpy(exec + 0x300, lits, sizeof(lits));
    memcpy(exec + 0x1000, tramp, 4 * n);
    uint32_t slot = (uint32_t)tsd_offset;
    memcpy(exec + 0x2000, &slot, 4);
    __builtin___clear_cache((char *)exec, (char *)exec + BLOB_SIZE);
    fill(teb, TEB_SIZE, seed);
    fill(scratch, SCRATCH_SIZE, seed * 3);
    memset(tsd, 0, 4096);
    tsd[tsd_offset / 8] = (uint64_t)(uintptr_t)(teb + 0x8000);
    memcpy(in, regs, 32 * 8);
    in[32] = (uint64_t)(uintptr_t)tsd | 5;           // the low 3 bits are a CPU number
    memset(out_regs, 0, sizeof(out_regs));
    run_case(in, exec);
}

int main(void)
{
    uint32_t count;
    read_exact(&count, 4);
    read_exact(common, sizeof(common));
    uint8_t *exec = mmap(NULL, BLOB_SIZE, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    uint64_t *tsd = aligned_alloc(4096, 4096);
    uint8_t *teb = aligned_alloc(4096, TEB_SIZE), *scratch = aligned_alloc(4096, SCRATCH_SIZE);
    uint8_t *teb1 = malloc(TEB_SIZE), *scratch1 = malloc(SCRATCH_SIZE);
    if (exec == MAP_FAILED || !tsd || !teb || !scratch || !teb1 || !scratch1) return 2;
    for (uint32_t c = 0; c < count; c++) {
        uint32_t original[2], rewritten[2], n, tramp[(0x2000 - 0x1000) / 4];
        uint64_t regs[32], tsd_offset, seed, before[33];
        read_exact(original, 8);
        read_exact(rewritten, 8);
        read_exact(&n, 4);
        if (n > sizeof(tramp) / 4) return 2;
        read_exact(tramp, 4 * n);
        read_exact(regs, sizeof(regs));
        read_exact(&tsd_offset, 8);
        read_exact(&seed, 8);
        if (tsd_offset > 4096 - 8 || tsd_offset % 8) return 2;
        for (int i = 0; i < 31; i++) {                // 'T'/'S' in the top byte: a pointer into the TEB or scratch memory
            if (regs[i] >> 56 == 'T') regs[i] = (uint64_t)(uintptr_t)(teb + 0x8000) + (regs[i] & 0xFFFFFF) - 0x800000;
            if (regs[i] >> 56 == 'S') regs[i] = (uint64_t)(uintptr_t)(scratch + 0x8000) + (regs[i] & 0xFFFFFF) - 0x800000;
        }
        uint64_t x18 = regs[18];
        run(exec, original, tramp, 0, regs, tsd_offset, seed, tsd, teb, scratch);
        memcpy(before, out_regs, sizeof(before));
        memcpy(teb1, teb, TEB_SIZE);
        memcpy(scratch1, scratch, SCRATCH_SIZE);
        regs[18] = 0x5eb0000000000018ull;            // the rewritten code must not read x18
        run(exec, rewritten, tramp, n, regs, tsd_offset, seed, tsd, teb, scratch);
        char why[256] = "";
        for (int i = 0; i < 32 && !why[0]; i++)
            if (i != 18 && before[i] != out_regs[i])
                snprintf(why, sizeof(why), "%s%d: %llx, rewritten %llx", i == 31 ? "nzcv" : "x", i == 31 ? 0 : i,
                         (unsigned long long)before[i], (unsigned long long)out_regs[i]);
        if (!why[0] && out_regs[18] != regs[18])
            snprintf(why, sizeof(why), "x18 changed: %llx", (unsigned long long)out_regs[18]);
        if (!why[0] && before[18] != x18)
            snprintf(why, sizeof(why), "the original changed x18");
        if (!why[0] && memcmp(teb, teb1, TEB_SIZE))
            snprintf(why, sizeof(why), "TEB memory differs");
        if (!why[0] && memcmp(scratch, scratch1, SCRATCH_SIZE))
            snprintf(why, sizeof(why), "memory differs");
        printf("%s\n", why[0] ? why : "ok");
    }
    return 0;
}
