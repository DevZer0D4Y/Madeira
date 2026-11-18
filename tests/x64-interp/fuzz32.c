// SPDX-License-Identifier: GPL-3.0-or-later
// Madeira Converter Exception: see LICENSE-EXCEPTION.md
//
// The 32-bit half of the differential test: a 32-bit Linux process (gcc -m32)
// runs each snippet from gen-fuzz-insns32.py natively, from random registers,
// flags, x87/MMX/SSE state (FXRSTOR) and memory, and records the results
// (FXSAVE and the registers). `fuzz replay` then runs the same snippets
// through X64Interpreter's 32-bit mode from the same inputs and compares.
//
// Usage: fuzz32 FILE TRIALS SEED

#define _GNU_SOURCE
#include "fuzz-common.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

extern const struct Template fuzz_templates[];
extern const uint32_t fuzz_template_count;
extern const uint8_t stub_pre_start[], stub_pre_end[], stub_post_start[], stub_post_end[];

static int run_native(void)
{
    uint32_t host_mxcsr;
    __asm__ volatile("stmxcsr %0" : "=m"(host_mxcsr));
    // Through a pointer: a direct call to a fixed address in a PIE comes out
    // relative to where the program was loaded.
    void (*volatile snippet)(void) = (void (*)(void))CODE;
    fault_signal = 0;
    if (sigsetjmp(fault_jump, 1) == 0) snippet();
    __asm__ volatile("fninit\n\temms\n\tldmxcsr %0" : : "m"(host_mxcsr));
    return fault_signal;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: fuzz32 FILE TRIALS SEED\n");
        return 2;
    }
    if (!map_fixed(CODE, 0x10000, PROT_READ | PROT_WRITE | PROT_EXEC) || !map_fixed(ANCHOR, 0x1000, PROT_READ | PROT_WRITE) ||
        !map_fixed(MEM, MEMSZ, PROT_READ | PROT_WRITE) || !map_fixed(STACK, STACKSZ, PROT_READ | PROT_WRITE)) {
        perror("mmap");
        return 2;
    }
    catch_faults();
    long trials = atol(argv[2]);
    rng = strtoull(argv[3], NULL, 0);
    if (!rng) rng = 1;
    FILE *f = fopen(argv[1], "wb");
    if (!f) { perror(argv[1]); return 2; }

    uint16_t sel[6];   // cs ds es ss fs gs, as the replay's state starts
    __asm__ volatile("mov %%cs, %0" : "=r"(sel[0]));
    __asm__ volatile("mov %%ds, %0" : "=r"(sel[1]));
    __asm__ volatile("mov %%es, %0" : "=r"(sel[2]));
    __asm__ volatile("mov %%ss, %0" : "=r"(sel[3]));
    __asm__ volatile("mov %%fs, %0" : "=r"(sel[4]));
    __asm__ volatile("mov %%gs, %0" : "=r"(sel[5]));
    size_t pre = (size_t)(stub_pre_end - stub_pre_start), post = (size_t)(stub_post_end - stub_post_start);
    fwrite("X86FUZZ1", 1, 8, f);
    uint64_t header[4] = { fuzz_template_count, (uint64_t)trials, rng, pre };
    fwrite(header, sizeof header, 1, f);
    fwrite(sel, sizeof sel, 1, f);

    long runs = 0, faults = 0;
    for (uint32_t n = 0; n < fuzz_template_count; n++) {
        const struct Template *t = &fuzz_templates[n];
        size_t len = (size_t)(t->end - t->start);
        uint8_t *code = (uint8_t *)CODE;
        memcpy(code, stub_pre_start, pre);
        memcpy(code + pre, t->start, len);
        memcpy(code + pre + len, stub_post_start, post);
        uint32_t name_len = (uint32_t)strlen(t->name), code_len = (uint32_t)len;
        fwrite(&name_len, 4, 1, f); fwrite(t->name, 1, name_len, f);
        fwrite(&code_len, 4, 1, f); fwrite(t->start, 1, len, f);
        fwrite(&t->undef, 8, 1, f); fwrite(&t->flags, 8, 1, f);
        for (long trial = 0; trial < trials; trial++) {
            setup_input32(t->flags);
            int fault = run_native();
            uint64_t hashes[2] = { hash((void *)MEM, MEMSZ), hash((void *)STACK, STACKSZ) };
            uint8_t faulted = (uint8_t)fault;
            fwrite(OUT32, sizeof *OUT32, 1, f);
            fwrite(hashes, sizeof hashes, 1, f);
            fwrite(&faulted, 1, 1, f);
            runs++;
            if (fault) {
                if (faults < 20) fprintf(stderr, "NATIVE FAULT %d in [%s]\n", fault, t->name);
                faults++;
            }
        }
    }
    fclose(f);
    printf("%u snippets, %ld recorded, %ld native faults\n", fuzz_template_count, runs, faults);
    return faults ? 1 : 0;
}
