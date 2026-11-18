// SPDX-License-Identifier: GPL-3.0-or-later
// Madeira Converter Exception: see LICENSE-EXCEPTION.md
//
// The engine check's 32-bit programs (../workloads.c) as a 32-bit Windows
// program, for tests/host/check-wow64interp-wine.py: each runs with each seed
// and prints its result, which must not depend on the CPU module.

#include "../workloads.c"

#include <stdio.h>
#include <string.h>

#include "NoJITSelfCheck.h"

static uint64_t host_callback(uint64_t a, uint64_t b)
{
    return madeira_nojit_callback(a, b);
}

typedef uint64_t (*entry_fn)(uint64_t, uint8_t *, callback_fn);

static const struct {
    const char *name;
    entry_fn entry;
} entries[] = {
    { "crc32", wl_crc32 }, { "sort", wl_sort }, { "sieve", wl_sieve }, { "divide", wl_divide },
    { "float", wl_float }, { "mmx", wl_mmx }, { "x87", wl_x87 }, { "strings", wl_strings },
    { "dispatch", wl_dispatch }, { "callback", wl_callback }, { "bits", wl_bits }, { "atomic", wl_atomic },
    { "rounding", wl_rounding },
};

static uint8_t buffer[65536];

int main(void)
{
    static const uint64_t seeds[] = { 0x1, 0x5eed1234cafef00dull, 0x0123456789abcdefull };
    for (unsigned i = 0; i < sizeof entries / sizeof *entries; i++) {
        for (unsigned j = 0; j < sizeof seeds / sizeof *seeds; j++) {
            memset(buffer, 0, sizeof buffer);
            uint64_t result = entries[i].entry(seeds[j], buffer, host_callback);
            printf("%s %llx %llx\n", entries[i].name, (unsigned long long)seeds[j], (unsigned long long)result);
        }
    }
    return 0;
}
