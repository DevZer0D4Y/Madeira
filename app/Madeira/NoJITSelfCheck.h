// SPDX-License-Identifier: GPL-3.0-or-later
// Madeira Converter Exception: see LICENSE-EXCEPTION.md
//
// The No JIT engine check: runs a set of x86-64 programs through
// X64Interpreter, from ordinary (non-executable) memory, and compares each
// result with what the same bytes gave on a real x86-64 CPU. The programs and
// their results are in NoJITSelfCheckData.h, made by
// tests/x64-interp/gen-selfcheck.py from tests/x64-interp/workloads.c.

#ifndef NOJIT_SELF_CHECK_H
#define NOJIT_SELF_CHECK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MadeiraNoJITReport {
    int passed;
    int total;
    uint64_t instructions;      // guest instructions interpreted
    double seconds;             // time spent interpreting
    uint64_t callbacks;         // calls the guest made into host (native) code
} MadeiraNoJITReport;

/// Runs every program; true when all of them give their expected result. `text`
/// (may be NULL) receives one line per program, then a summary, NUL-terminated.
/// Takes a second or two; call it off the main thread.
bool madeira_nojit_self_check(MadeiraNoJITReport *report, char *text, size_t text_size);

/// The host function the "callback" program calls: native code reached from
/// interpreted code, as Wine's own functions will be. The native reference run
/// calls the same function.
static inline uint64_t madeira_nojit_callback(uint64_t a, uint64_t b)
{
    uint64_t h = a * 0x9e3779b97f4a7c15ull ^ (b + 0x632be59bd9b4e019ull);
    return (h << 29) | (h >> 35);
}

#ifdef __cplusplus
}
#endif

#endif // NOJIT_SELF_CHECK_H
