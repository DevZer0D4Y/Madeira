// SPDX-License-Identifier: GPL-3.0-or-later
// Madeira Converter Exception: see LICENSE-EXCEPTION.md
//
// Runs the self-check programs natively on an x86-64 host: the reference
// results gen-selfcheck.py records. Linux or macOS on x86-64.
//
// Usage: run-native BLOB NAME:OFFSET:SEED...   prints NAME SEED RESULT per line

#define _GNU_SOURCE
#include "NoJITSelfCheck.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

typedef uint64_t (__attribute__((ms_abi)) *callback_fn)(uint64_t, uint64_t);
typedef uint64_t (__attribute__((ms_abi)) *entry_fn)(uint64_t, void *, callback_fn);

static uint64_t __attribute__((ms_abi)) callback(uint64_t a, uint64_t b)
{
    return madeira_nojit_callback(a, b);
}

int main(int argc, char **argv)
{
    if (argc < 3) return 2;
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 2; }
    uint8_t *code = mmap(NULL, 1 << 20, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (code == MAP_FAILED) { perror("mmap"); return 2; }
    size_t size = fread(code, 1, 1 << 20, f);
    fclose(f);
    __builtin___clear_cache((char *)code, (char *)code + size);
    void *scratch = aligned_alloc(4096, 64 * 1024);
    for (int i = 2; i < argc; i++) {
        char name[64];
        unsigned long long offset, seed;
        if (sscanf(argv[i], "%63[^:]:%llx:%llx", name, &offset, &seed) != 3) return 2;
        memset(scratch, 0, 64 * 1024);
        uint64_t result = ((entry_fn)(code + offset))(seed, scratch, callback);
        printf("%s %llx %llx\n", name, seed, (unsigned long long)result);
    }
    return 0;
}
