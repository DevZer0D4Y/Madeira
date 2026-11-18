// SPDX-License-Identifier: GPL-3.0-or-later
// Madeira Converter Exception: see LICENSE-EXCEPTION.md

#if !defined(__APPLE__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L   // posix_memalign and clock_gettime under -std=c11
#endif

#include "NoJITSelfCheck.h"
#include "NoJITSelfCheckData.h"
#include "X64Interpreter.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Addresses the guest jumps to but that are never fetched from: the interpreter
// stops at the first and hands the second to the host.
#define STOP_ADDRESS     0x1000u
#define CALLBACK_ADDRESS 0x2000u
#define SCRATCH_SIZE     (64u * 1024u)
#define STACK_SIZE       (1024u * 1024u)
#define BUDGET           2000000000ull

static bool is_callback(void *user, uint64_t target)
{
    (void)user;
    return target == CALLBACK_ADDRESS;
}

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef struct Text { char *buffer; size_t size, used; } Text;

static void append(Text *text, const char *line)
{
    if (!text->buffer || text->used + 1 >= text->size) return;
    int n = snprintf(text->buffer + text->used, text->size - text->used, "%s\n", line);
    if (n > 0) text->used += (size_t)n < text->size - text->used ? (size_t)n : text->size - text->used - 1;
}

// The 32-bit programs run in a 4 MiB guest window, as Madeira's WoW64 code
// does: guest address a is host address window + a.
#define WINDOW_SIZE      (4u * 1024u * 1024u)
#define GUEST_CODE       0x10000u
#define GUEST_SCRATCH    0x100000u
#define GUEST_STACK_TOP  0x3f0000u

static uint64_t read64(const void *p) { uint64_t v; memcpy(&v, p, 8); return v; }

bool madeira_nojit_self_check(MadeiraNoJITReport *report, char *text_buffer, size_t text_size)
{
    memset(report, 0, sizeof *report);
    Text out = { text_buffer, text_size, 0 };
    if (text_buffer && text_size) text_buffer[0] = 0;
    size_t count = sizeof nojit_selfcheck_cases / sizeof nojit_selfcheck_cases[0];
    size_t code_size = (sizeof nojit_selfcheck_blob64 + 4095) & ~(size_t)4095;
    void *code = NULL, *scratch = NULL, *stack = NULL, *window = NULL;
    report->total = (int)count;
    if (posix_memalign(&code, 4096, code_size) || posix_memalign(&scratch, 4096, SCRATCH_SIZE) ||
        posix_memalign(&stack, 4096, STACK_SIZE) || posix_memalign(&window, 4096, WINDOW_SIZE)) {
        append(&out, "Could not allocate memory for the check.");
        free(code); free(scratch); free(stack); free(window);
        return false;
    }
    // Plain heap memory: readable and writable, never executable. The interpreter
    // only reads the code as data.
    memcpy(code, nojit_selfcheck_blob64, sizeof nojit_selfcheck_blob64);
    uint8_t *w = (uint8_t *)window;
    memset(w, 0, WINDOW_SIZE);
    memcpy(w + GUEST_CODE, nojit_selfcheck_blob32, sizeof nojit_selfcheck_blob32);

    char line[256];
    for (size_t i = 0; i < count; i++) {
        const struct NoJITCase *c = &nojit_selfcheck_cases[i];
        bool m32 = c->bits == 32;
        X64State s;
        uint64_t stop = STOP_ADDRESS;
        if (m32) {
            // cdecl: the stop address as the return address, then seed (two halves), buffer, callback.
            memset(w + GUEST_SCRATCH, 0, SCRATCH_SIZE);
            x64_state_init32(&s, (uint64_t)(uintptr_t)w);
            uint32_t esp = GUEST_STACK_TOP - 32;
            uint32_t frame[5] = { STOP_ADDRESS, (uint32_t)c->seed, (uint32_t)(c->seed >> 32), GUEST_SCRATCH, CALLBACK_ADDRESS };
            memcpy(w + esp, frame, sizeof frame);
            s.gpr[X64_RSP] = esp;
            s.rip = GUEST_CODE + c->entry;
        } else {
            memset(scratch, 0, SCRATCH_SIZE);
            x64_state_init(&s);
            // Windows x64 entry: 32 bytes of home space above the return address, which
            // is the stop address, and RSP 8 below a 16-byte boundary.
            uint64_t rsp = (((uint64_t)(uintptr_t)stack + STACK_SIZE) & ~(uint64_t)15) - 64;
            rsp -= 8;
            memcpy((void *)(uintptr_t)rsp, &stop, 8);
            s.gpr[X64_RSP] = rsp;
            s.gpr[X64_RCX] = c->seed;
            s.gpr[X64_RDX] = (uint64_t)(uintptr_t)scratch;
            s.gpr[X64_R8] = CALLBACK_ADDRESS;
            s.rip = (uint64_t)(uintptr_t)code + c->entry;
        }
        X64Env env = { .stop_rip = STOP_ADDRESS, .is_native = is_callback, .user = NULL };
        X64ExitInfo info;
        X64Exit exit;
        uint64_t callbacks = 0;
        double start = now();
        for (;;) {
            exit = x64_run(&s, &env, BUDGET, &info);
            if (exit != X64_EXIT_NATIVE || s.rip != CALLBACK_ADDRESS) break;
            callbacks++;
            uint64_t a, b;
            if (m32) {   // cdecl (uint64, uint64) above the return address
                const uint8_t *args = w + (uint32_t)s.gpr[X64_RSP] + 4;
                a = read64(args);
                b = read64(args + 8);
            } else {
                a = s.gpr[X64_RCX];
                b = s.gpr[X64_RDX];
            }
            x64_return_from_native(&s, madeira_nojit_callback(a, b));
        }
        double seconds = now() - start;
        report->seconds += seconds;
        report->instructions += s.instructions;
        report->callbacks += callbacks;
        uint64_t result = m32 ? (s.gpr[X64_RDX] << 32) | (uint32_t)s.gpr[X64_RAX] : s.gpr[X64_RAX];

        if (exit == X64_EXIT_STOP && result == c->expect) {
            report->passed++;
            snprintf(line, sizeof line, "ok   %d-bit %-9s seed %llx: %llu instructions in %.0f ms", c->bits, c->name,
                     (unsigned long long)c->seed, (unsigned long long)s.instructions, seconds * 1000);
        } else if (exit == X64_EXIT_STOP) {
            snprintf(line, sizeof line, "FAIL %d-bit %-9s seed %llx: got %016llx, expected %016llx", c->bits, c->name,
                     (unsigned long long)c->seed, (unsigned long long)result, (unsigned long long)c->expect);
        } else {
            uint64_t base = m32 ? GUEST_CODE : (uint64_t)(uintptr_t)code;
            int k = snprintf(line, sizeof line, "FAIL %d-bit %-9s seed %llx: stopped (%s) at +%llx:", c->bits, c->name,
                             (unsigned long long)c->seed, x64_exit_name(exit), (unsigned long long)(info.rip - base));
            for (int b = 0; b < info.length && k < (int)sizeof line - 4; b++)
                k += snprintf(line + k, sizeof line - (size_t)k, " %02x", info.bytes[b]);
        }
        append(&out, line);
    }
    snprintf(line, sizeof line, "%d of %d checks matched a real x86 CPU (64- and 32-bit): %llu instructions, %llu host calls, "
             "%.1f million instructions a second.", report->passed, report->total,
             (unsigned long long)report->instructions, (unsigned long long)report->callbacks,
             report->seconds > 0 ? (double)report->instructions / report->seconds / 1e6 : 0.0);
    append(&out, line);
    free(code); free(scratch); free(stack); free(window);
    return report->passed == report->total;
}
