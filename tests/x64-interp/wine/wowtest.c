// SPDX-License-Identifier: GPL-3.0-or-later
// Madeira Converter Exception: see LICENSE-EXCEPTION.md
//
// A 32-bit Windows program for testing a WoW64 CPU module in Wine
// (tests/host/check-wow64interp-wine.py). It exercises what a 32-bit game
// needs from the CPU module: system calls, unix calls, exceptions raised by the
// CPU and resumed from a changed context, threads and atomics, x87 floating
// point through msvcrt, and file and socket I/O.
//
// Run once with Wine's wow64cpu.dll (the real CPU) and once with
// wow64interp.dll: every line not starting with "info:" must be the same.

#include <winsock2.h>
#include <windows.h>
#include <cpuid.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Exceptions

static void *recover_buf[5];
static volatile DWORD seen_code, seen_address, seen_eip, seen_params, seen_info0, seen_info1, seen_xmm0, seen_flags;
static volatile int handled;

static __attribute__((noinline, noreturn)) void recover(void)
{
    __builtin_longjmp(recover_buf, 1);
}

static LONG CALLBACK handler(EXCEPTION_POINTERS *pointers)
{
    EXCEPTION_RECORD *record = pointers->ExceptionRecord;
    CONTEXT *context = pointers->ContextRecord;
    if (record->ExceptionCode == DBG_PRINTEXCEPTION_C || record->ExceptionCode == 0x4001000a) return EXCEPTION_CONTINUE_SEARCH;
    handled++;
    seen_code = record->ExceptionCode;
    seen_address = (DWORD)record->ExceptionAddress;
    seen_eip = context->Eip;
    seen_params = record->NumberParameters;
    seen_info0 = record->NumberParameters > 0 ? (DWORD)record->ExceptionInformation[0] : 0xdead;
    seen_info1 = record->NumberParameters > 1 ? (DWORD)record->ExceptionInformation[1] : 0xdead;
    seen_flags = context->ContextFlags;
    memcpy((void *)&seen_xmm0, context->ExtendedRegisters + 160, 4);   // FXSAVE: XMM0
    if (record->ExceptionCode == EXCEPTION_BREAKPOINT) {
        context->Eip += 1;   // past the INT3, and on
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    if (record->ExceptionCode == 0xe0000001) return EXCEPTION_CONTINUE_EXECUTION;
    context->Eip = (DWORD)recover;
    return EXCEPTION_CONTINUE_EXECUTION;
}

extern char fault_read[], fault_write[], fault_divide[], fault_int3[], fault_ud2[], fault_overflow[];

// Each loads a marker into XMM0 first: the handler's context must show the
// 32-bit program's XMM0, not the CPU module's.
#define FAULT(name, label, insns)                                                               \
    static __attribute__((noinline)) unsigned name(void)                                        \
    {                                                                                           \
        unsigned result;                                                                        \
        __asm__ volatile("movl $0x5a5a1234, %%eax\n\tmovd %%eax, %%xmm0\n\txorl %%eax, %%eax\n\t" \
                         ".globl _" #label "\n_" #label ":\n\t" insns                           \
                         : "=a"(result) : : "ecx", "edx", "xmm0", "memory");                    \
        return result;                                                                          \
    }

FAULT(do_read, fault_read, "movl 0x0, %%eax")
FAULT(do_write, fault_write, "movl $1, 0x10")
FAULT(do_divide, fault_divide, "xorl %%ecx, %%ecx\n\tmovl $7, %%eax\n\tcltd\n\tidivl %%ecx")
FAULT(do_int3, fault_int3, "int3\n\tmovl $77, %%eax")
FAULT(do_ud2, fault_ud2, "ud2")
FAULT(do_overflow, fault_overflow, "movl $0x7fffffff, %%eax\n\taddl $1, %%eax\n\tinto")

static unsigned recursion(unsigned n)
{
    volatile char pad[512];
    pad[0] = (char)n;
    return n == 0 ? (unsigned)pad[0] : recursion(n - 1) + (unsigned)pad[0];
}

static void report(const char *name, const void *label, unsigned result)
{
    printf("exception %-9s code %08lx at+%ld eip+%ld params %lu %lx %lx xmm0 %08lx extended %d result %u handled %d\n", name,
           (unsigned long)seen_code, (long)(seen_address - (DWORD)label), (long)(seen_eip - (DWORD)label),
           (unsigned long)seen_params, (unsigned long)seen_info0, (unsigned long)seen_info1, (unsigned long)seen_xmm0,
           (seen_flags & CONTEXT_EXTENDED_REGISTERS) == CONTEXT_EXTENDED_REGISTERS, result, handled);
    seen_code = seen_address = seen_eip = seen_params = seen_info0 = seen_info1 = seen_xmm0 = seen_flags = 0;
    handled = 0;
}

#define TRY(name, label, call)                                    \
    do {                                                          \
        volatile unsigned result = 0;                             \
        if (!__builtin_setjmp(recover_buf)) result = call;        \
        report(name, label, result);                              \
    } while (0)

static void exceptions(void)
{
    PVOID vectored = AddVectoredExceptionHandler(1, handler);
    TRY("read", fault_read, do_read());
    TRY("write", fault_write, do_write());
    TRY("divide", fault_divide, do_divide());
    TRY("int3", fault_int3, do_int3());
    TRY("ud2", fault_ud2, do_ud2());
    TRY("into", fault_overflow, do_overflow());
    TRY("raise", 0, (RaiseException(0xe0000001, 0, 0, NULL), 5));
    // The 32-bit stack's guard pages: deep recursion grows it, deeper overflows it.
    TRY("deep", 0, recursion(400));
    TRY("overflow", 0, recursion(100000000));
    RemoveVectoredExceptionHandler(vectored);
}

// ---------------------------------------------------------------------------
// Threads and atomics

static volatile LONG counter;
static volatile LONG spin;
static volatile uint64_t shared_sum;
static volatile int64_t wide;

static DWORD WINAPI worker(void *arg)
{
    unsigned id = (unsigned)(uintptr_t)arg;
    uint32_t local = id;
    for (int i = 0; i < 100000; i++) {
        InterlockedIncrement(&counter);
        local = local * 1664525u + 1013904223u;
        while (InterlockedCompareExchange(&spin, 1, 0)) YieldProcessor();   // a spin lock around a plain sum
        shared_sum += local >> 16;
        InterlockedExchange(&spin, 0);
        int64_t old, new;
        do {   // CMPXCHG8B
            old = wide;
            new = old + 3;
        } while (__sync_val_compare_and_swap(&wide, old, new) != old);
    }
    return local;
}

static void threads(void)
{
    HANDLE handles[4];
    DWORD codes[4];
    for (int i = 0; i < 4; i++) handles[i] = CreateThread(NULL, 0, worker, (void *)(uintptr_t)(i + 1), 0, NULL);
    WaitForMultipleObjects(4, handles, TRUE, INFINITE);
    for (int i = 0; i < 4; i++) {
        GetExitCodeThread(handles[i], &codes[i]);
        CloseHandle(handles[i]);
    }
    printf("threads counter %ld wide %lld sum %llu exits %08lx %08lx %08lx %08lx\n", (long)counter, (long long)wide,
           (unsigned long long)shared_sum, codes[0], codes[1], codes[2], codes[3]);
}

// ---------------------------------------------------------------------------
// Floating point and integer arithmetic through msvcrt and libgcc

static int compare(const void *a, const void *b)
{
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

static void arithmetic(void)
{
    static const double values[] = { 0.5, 1.0, 2.5, 10.0, 1e-3, 123.456, 1e10, -3.75 };
    for (unsigned i = 0; i < sizeof values / sizeof *values; i++) {
        double x = values[i];
        printf("math %.17g: %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n", x, sin(x), cos(x), tan(x),
               sqrt(fabs(x)), exp(x / 8), log(fabs(x)), pow(fabs(x), 1.7), atan2(x, 3.0), fmod(x, 0.7));
    }
    float f = 1.0f;
    long double l = 1.0L;
    for (int i = 1; i < 40; i++) {
        f = f * 1.37f + 0.11f / (float)i;
        l = l * 1.37L + 0.11L / (long double)i;
    }
    printf("float %.9g long double %.17Lg int %d %lld\n", f, l, (int)f, (long long)l);
    char text[64];
    double parsed = strtod("3.14159265358979323846e-7", NULL);
    snprintf(text, sizeof text, "%.15e|%g|%10.4f|%a", parsed, parsed * 1e9, -parsed * 1e7, 0.1);
    printf("format %s\n", text);
    uint64_t a = 0x123456789abcdef1ull, b = 0x2468ace1ull;
    int64_t c = -0x7edcba9876543210ll, d = 0x13579bdfll;
    printf("int64 %llx %llx %lld %lld %llx\n", (unsigned long long)(a / b), (unsigned long long)(a % b), (long long)(c / d),
           (long long)(c % d), (unsigned long long)(a * b));
    int sorted[1000];
    uint32_t x = 12345;
    for (int i = 0; i < 1000; i++) sorted[i] = (int)((x = x * 1103515245u + 12345u) >> 1);
    qsort(sorted, 1000, sizeof *sorted, compare);
    uint32_t hash = 0;
    for (int i = 0; i < 1000; i++) hash = hash * 31 + (uint32_t)sorted[i];
    printf("qsort %08x first %d last %d\n", hash, sorted[0], sorted[999]);
}

// ---------------------------------------------------------------------------
// I/O: files (system calls) and sockets (system and unix calls)

static void files(void)
{
    char folder[MAX_PATH], path[MAX_PATH];
    GetTempPathA(sizeof folder, folder);
    GetTempFileNameA(folder, "wow", 0, path);
    static unsigned char data[300000], back[300000];
    for (unsigned i = 0; i < sizeof data; i++) data[i] = (unsigned char)(i * 7 + (i >> 9));
    HANDLE file = CreateFileA(path, GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, 0, NULL);
    DWORD written = 0, read = 0;
    BOOL ok = WriteFile(file, data, sizeof data, &written, NULL);
    SetFilePointer(file, 0, NULL, FILE_BEGIN);
    ok = ok && ReadFile(file, back, sizeof back, &read, NULL);
    CloseHandle(file);
    DeleteFileA(path);
    printf("files %d written %lu read %lu same %d\n", ok, written, read, !memcmp(data, back, sizeof data));
}

static void sockets(void)
{
    WSADATA wsa;
    int started = WSAStartup(MAKEWORD(2, 2), &wsa);
    char name[256] = "";
    int named = gethostname(name, sizeof name);
    SOCKET s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    struct sockaddr_in address = { 0 };
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    int length = sizeof address;
    int bound = bind(s, (struct sockaddr *)&address, sizeof address);
    getsockname(s, (struct sockaddr *)&address, &length);
    const char message[] = "a datagram through the CPU module";
    int sent = sendto(s, message, sizeof message, 0, (struct sockaddr *)&address, sizeof address);
    char reply[64] = "";
    int received = recv(s, reply, sizeof reply, 0);
    struct hostent *host = gethostbyname("localhost");
    printf("sockets started %d named %d nonempty %d bound %d sent %d received %d same %d localhost %d\n", started, named,
           name[0] != 0, bound, sent, received, !strcmp(reply, message), host != NULL);
    closesocket(s);
    WSACleanup();
}

// ---------------------------------------------------------------------------
// What differs between CPU modules: printed, not compared

static void info(void)
{
    unsigned eax, ebx, ecx, edx;
    char vendor[13] = "";
    __cpuid(0, eax, ebx, ecx, edx);
    memcpy(vendor, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    __cpuid(1, eax, ebx, ecx, edx);
    printf("info: cpuid %s leaf1 ecx %08x edx %08x; sse2 %d sse3 %d\n", vendor, ecx, edx,
           IsProcessorFeaturePresent(PF_XMMI64_INSTRUCTIONS_AVAILABLE), IsProcessorFeaturePresent(PF_SSE3_INSTRUCTIONS_AVAILABLE));
    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    volatile uint32_t sink = 0;
    uint32_t v = 1;
    for (int i = 0; i < 20000000; i++) v = v * 1664525u + 1013904223u;   // about 4 instructions an iteration
    sink = v;
    QueryPerformanceCounter(&end);
    double seconds = (double)(end.QuadPart - start.QuadPart) / (double)frequency.QuadPart;
    printf("info: 20M loop iterations in %.3f s (%u)\n", seconds, (unsigned)sink);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("start\n");
    exceptions();
    threads();
    arithmetic();
    files();
    sockets();
    info();
    printf("done\n");
    return 0;
}
