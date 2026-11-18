// SPDX-License-Identifier: GPL-3.0-or-later
// Madeira Converter Exception: see LICENSE-EXCEPTION.md
//
// A WoW64 CPU module that runs 32-bit x86 code with X64Interpreter instead of a
// JIT: the No JIT path for 32-bit Windows programs (docs/NO-JIT.md).
//
// Wine's wow64.dll loads one CPU module per 32-bit process (Wine's wow64cpu.dll
// on x86-64, FEX's xtajit.dll on ARM64) and drives it through the BTCpu*
// exports below. This one keeps the 32-bit registers in the thread's WoW64 CPU
// area whenever 32-bit code is not running, as wow64cpu does, so wow64.dll and
// ntdll read and change them there. It imports nothing but ntdll and wow64, as
// a module on the 64-bit side of a WoW64 process must.
//
// How 32-bit code leaves for the 64-bit side: wow64.dll points i386 ntdll's
// system call and unix call dispatchers at two addresses this module hands out
// (BTCpuGetBopCode, __wine_get_unix_opcode). Nothing executes them: the
// interpreter stops when a CALL or JMP reaches either, and the module makes the
// call on the 64-bit side, as wow64cpu's far jumps do.
//
// Faults: a 32-bit access to a bad address faults inside the interpreter, in
// 64-bit code. ntdll's 64-bit dispatcher calls BTCpuResetToConsistentState,
// which writes the 32-bit state (EIP at the faulting instruction) into the CPU
// area, and wow64.dll's handler around BTCpuSimulate passes the exception to
// the 32-bit side.
//
// Madeira's guest window: guest address a is host address B + a, where B comes
// from NtQueryInformationProcess(ProcessWineIosWowGuestBase); on upstream Wine
// the query fails and B is 0.

#include <stdint.h>
#include <string.h>

#include <windows.h>
#include <winternl.h>

#include "X64Interpreter.h"

#define WOW64_TLS_CPURESERVED 1
#define WOW64_TLS_WOW64INFO 10
#define WOW64_CPURESERVED_FLAG_RESET_STATE 1
#define WOW64_CPUFLAGS_SOFTWARE 0x02
#define ProcessWineIosWowGuestBase 1010   // Madeira's ntdll (docs/WOW64.md)

#ifndef STATUS_SUCCESS
#define STATUS_SUCCESS ((NTSTATUS)0)
#endif
#ifndef STATUS_NOT_SUPPORTED
#define STATUS_NOT_SUPPORTED ((NTSTATUS)0xC00000BB)
#endif
#ifndef STATUS_ASSERTION_FAILURE
#define STATUS_ASSERTION_FAILURE ((NTSTATUS)0xC0000420)
#endif
#ifndef STATUS_STACK_BUFFER_OVERRUN
#define STATUS_STACK_BUFFER_OVERRUN ((NTSTATUS)0xC0000409)
#endif

// The CPU area wow64 gives each thread: flags, machine, then the 32-bit context.
#pragma pack(push, 2)
typedef struct {
    USHORT Flags;
    USHORT Machine;
    WOW64_CONTEXT Context;
} CPU_AREA;
#pragma pack(pop)

typedef struct {
    ULONG NativeSystemPageSize;
    ULONG CpuFlags;
    ULONG Wow64ExecuteFlags;
    ULONG unknown;
    ULONGLONG SectionHandle;
    ULONGLONG CrossProcessWorkList;
    USHORT NativeMachineType;
    USHORT EmulatedMachineType;
} WOW64INFO_;

// ntdll and wow64 exports mingw's headers do not declare.
NTSTATUS WINAPI Wow64SystemServiceEx(UINT number, UINT *args);
void WINAPI Wow64PassExceptionToGuest(EXCEPTION_POINTERS *pointers);
NTSTATUS WINAPI RtlWow64GetThreadContext(HANDLE thread, WOW64_CONTEXT *context);
NTSTATUS WINAPI RtlWow64SetThreadContext(HANDLE thread, const WOW64_CONTEXT *context);
BOOLEAN WINAPI RtlIsProcessorFeaturePresent(UINT feature);
NTSTATUS WINAPI LdrGetDllHandle(LPCWSTR path, ULONG flags, const UNICODE_STRING *name, HMODULE *module);
void *WINAPI RtlFindExportedRoutineByName(HMODULE module, const char *name);
NTSTATUS WINAPI LdrDisableThreadCalloutsForDll(HMODULE module);
NTSTATUS WINAPI NtAllocateVirtualMemory(HANDLE process, PVOID *address, ULONG_PTR zero_bits, SIZE_T *size, ULONG type, ULONG protect);
ULONG __cdecl DbgPrint(const char *format, ...);

#define CURRENT_PROCESS ((HANDLE)~(ULONG_PTR)0)   // CURRENT_PROCESS is kernel32's

typedef NTSTATUS (WINAPI *unix_call_fn)(UINT64 handle, unsigned int code, void *args);

static uint64_t guest_base;      // B
static uint32_t bop_syscall;     // guest addresses of the two ways out
static uint32_t bop_unix;
static unix_call_fn unix_call_dispatcher;

// The live interpreter state of each thread running 32-bit code, so the fault
// path finds it: the CPU area has a fixed size, so it lives here, by TEB.
#define MAX_THREADS 4096
struct thread_slot {
    void *volatile teb;
    X64State *volatile state;
    volatile LONG running;
};
static struct thread_slot slots[MAX_THREADS];

static struct thread_slot *thread_slot(BOOL create)
{
    void *teb = NtCurrentTeb();
    size_t start = ((uintptr_t)teb >> 12) % MAX_THREADS;
    for (size_t i = 0; i < MAX_THREADS; i++) {
        struct thread_slot *slot = &slots[(start + i) % MAX_THREADS];
        void *key = slot->teb;
        if (key == teb) return slot;
        if (!key) {
            if (!create) return NULL;
            if (__sync_bool_compare_and_swap(&slot->teb, NULL, teb) || slot->teb == teb) return slot;
        }
    }
    return NULL;
}

static CPU_AREA *cpu_area(void)
{
    return (CPU_AREA *)NtCurrentTeb()->TlsSlots[WOW64_TLS_CPURESERVED];
}

static inline void *host(uint32_t guest) { return (void *)(uintptr_t)(guest_base + guest); }
static inline uint32_t read32(uint32_t guest) { uint32_t v; memcpy(&v, host(guest), 4); return v; }
static inline uint64_t read64(uint32_t guest) { uint64_t v; memcpy(&v, host(guest), 8); return v; }

// The 32-bit TEB, which FS addresses, as a guest address.
static uint32_t teb32(void)
{
    LONG offset;
    memcpy(&offset, (const char *)NtCurrentTeb() + 0x180c, sizeof offset);   // TEB.WowTebOffset
    return (uint32_t)((uintptr_t)NtCurrentTeb() + offset - guest_base);
}

static void load_state(X64State *s, const WOW64_CONTEXT *c)
{
    x64_state_init32(s, guest_base);
    s->gpr[X64_RAX] = c->Eax; s->gpr[X64_RCX] = c->Ecx; s->gpr[X64_RDX] = c->Edx; s->gpr[X64_RBX] = c->Ebx;
    s->gpr[X64_RSP] = c->Esp; s->gpr[X64_RBP] = c->Ebp; s->gpr[X64_RSI] = c->Esi; s->gpr[X64_RDI] = c->Edi;
    s->rip = c->Eip;
    s->rflags = (c->EFlags & 0x00244fd5u) | 0x202;
    s->seg_cs = (uint16_t)c->SegCs; s->seg_ds = (uint16_t)c->SegDs; s->seg_es = (uint16_t)c->SegEs;
    s->seg_fs = (uint16_t)c->SegFs; s->seg_gs = (uint16_t)c->SegGs; s->seg_ss = (uint16_t)c->SegSs;
    x64_fxrstor(s, c->ExtendedRegisters);
    s->fs_base = teb32();
}

// FXSAVE's image to FNSAVE's (I386_CONTEXT.FloatSave), as wow64cpu does.
static void fnsave_from_fxsave(WOW64_FLOATING_SAVE_AREA *fpu, const BYTE *fx)
{
    USHORT cw, sw;
    memcpy(&cw, fx, 2);
    memcpy(&sw, fx + 2, 2);
    unsigned top = (sw >> 11) & 7;
    fpu->ControlWord = 0xffff0000u | cw;
    fpu->StatusWord = 0xffff0000u | sw;
    fpu->ErrorOffset = fpu->ErrorSelector = fpu->DataOffset = fpu->DataSelector = 0;
    fpu->Cr0NpxState = 0xffff0000u | sw;
    fpu->TagWord = 0xffff0000u;
    for (unsigned i = 0; i < 8; i++) {
        memcpy(&fpu->RegisterArea[10 * i], fx + 32 + 16 * i, 10);   // ST(i)
        unsigned physical = (i + top) & 7, tag;
        const BYTE *reg = fx + 32 + 16 * ((physical - top) & 7);
        USHORT exponent;
        uint64_t mantissa;
        memcpy(&mantissa, reg, 8);
        memcpy(&exponent, reg + 8, 2);
        if (!(fx[4] & (1u << physical))) tag = 3;
        else if ((exponent & 0x7fff) == 0x7fff) tag = 2;
        else if (!(exponent & 0x7fff)) tag = mantissa ? 2 : 1;
        else tag = (mantissa >> 63) ? 0 : 2;
        fpu->TagWord |= tag << (2 * physical);
    }
}

static void store_state(WOW64_CONTEXT *c, const X64State *s)
{
    c->ContextFlags = WOW64_CONTEXT_ALL;
    c->Eax = (DWORD)s->gpr[X64_RAX]; c->Ecx = (DWORD)s->gpr[X64_RCX]; c->Edx = (DWORD)s->gpr[X64_RDX];
    c->Ebx = (DWORD)s->gpr[X64_RBX]; c->Esp = (DWORD)s->gpr[X64_RSP]; c->Ebp = (DWORD)s->gpr[X64_RBP];
    c->Esi = (DWORD)s->gpr[X64_RSI]; c->Edi = (DWORD)s->gpr[X64_RDI];
    c->Eip = (DWORD)s->rip;
    c->EFlags = (DWORD)s->rflags;
    c->SegCs = s->seg_cs; c->SegDs = s->seg_ds; c->SegEs = s->seg_es;
    c->SegFs = s->seg_fs; c->SegGs = s->seg_gs; c->SegSs = s->seg_ss;
    x64_fxsave(s, c->ExtendedRegisters);
    fnsave_from_fxsave(&c->FloatSave, c->ExtendedRegisters);
}

static bool is_bop(void *user, uint64_t target)
{
    (void)user;
    return target == bop_syscall || target == bop_unix;
}

// An exception the interpreter found (INT3, an illegal instruction...), raised
// on the 32-bit side at `eip` (`address`: where the record says it happened).
static void raise_guest(X64State *s, CPU_AREA *cpu, NTSTATUS code, uint32_t eip, uint32_t address,
                        ULONG parameters, ULONG_PTR p0, ULONG_PTR p1)
{
    EXCEPTION_RECORD record;
    CONTEXT context;
    EXCEPTION_POINTERS pointers = { &record, &context };
    memset(&record, 0, sizeof record);
    memset(&context, 0, sizeof context);   // no 64-bit state for wow64 to copy
    s->rip = eip;
    store_state(&cpu->Context, s);
    record.ExceptionCode = (DWORD)code;
    record.ExceptionAddress = host(address);
    record.NumberParameters = parameters;
    record.ExceptionInformation[0] = p0;
    record.ExceptionInformation[1] = p1;
    Wow64PassExceptionToGuest(&pointers);   // points the 32-bit context at KiUserExceptionDispatcher
    cpu->Flags &= (USHORT)~WOW64_CPURESERVED_FLAG_RESET_STATE;
    load_state(s, &cpu->Context);
}

// After a call to the 64-bit side: wow64 or ntdll may have replaced the context
// (an APC, an exception, NtContinue, NtSetContextThread), which they mark.
static void returned(X64State *s, CPU_AREA *cpu, NTSTATUS status)
{
    cpu->Context.Eax = (DWORD)status;
    if (cpu->Flags & WOW64_CPURESERVED_FLAG_RESET_STATE) {
        cpu->Flags &= (USHORT)~WOW64_CPURESERVED_FLAG_RESET_STATE;
        load_state(s, &cpu->Context);
    } else {
        s->gpr[X64_RAX] = (uint32_t)status;
    }
}

void WINAPI BTCpuSimulate(void)
{
    CPU_AREA *cpu = cpu_area();
    struct thread_slot *slot = thread_slot(TRUE);
    X64State s;
    X64Env env = { .stop_rip = ~0ull, .is_native = is_bop, .user = NULL };
    load_state(&s, &cpu->Context);
    cpu->Flags &= (USHORT)~WOW64_CPURESERVED_FLAG_RESET_STATE;

    for (;;) {
        X64ExitInfo info;
        if (slot) { slot->state = &s; slot->running = 1; }   // a callback may have used the slot
        X64Exit exit = x64_run(&s, &env, UINT64_MAX, &info);
        if (slot) slot->running = 0;
        uint32_t eip = (uint32_t)info.rip, esp = (uint32_t)s.gpr[X64_RSP];

        switch (exit) {
        case X64_EXIT_NATIVE:
            if ((uint32_t)s.rip == bop_syscall) {
                // From an i386 ntdll stub: [esp] returns into the stub, the arguments start at esp + 8.
                s.rip = read32(esp);
                s.gpr[X64_RSP] = esp + 4;
                store_state(&cpu->Context, &s);
                returned(&s, cpu, Wow64SystemServiceEx((UINT)s.gpr[X64_RAX], (UINT *)host(esp + 8)));
                continue;
            }
            if ((uint32_t)s.rip == bop_unix && unix_call_dispatcher) {
                // __wine_unix_call(handle, code, args), stdcall: 16 bytes of arguments.
                uint64_t handle = read64(esp + 4);
                uint32_t code = read32(esp + 12), args = read32(esp + 16);
                s.rip = read32(esp);
                s.gpr[X64_RSP] = esp + 20;
                store_state(&cpu->Context, &s);
                returned(&s, cpu, unix_call_dispatcher(handle, code, args ? host(args) : NULL));
                continue;
            }
            raise_guest(&s, cpu, STATUS_ILLEGAL_INSTRUCTION, eip, eip, 0, 0, 0);
            continue;
        case X64_EXIT_BREAKPOINT:   // EIP after the INT3; wow64 steps it back (WOW64_CPUFLAGS_SOFTWARE)
            raise_guest(&s, cpu, info.length == 1 && info.bytes[0] == 0xF1 ? STATUS_SINGLE_STEP : STATUS_BREAKPOINT,
                        (uint32_t)s.rip, eip, 1, 0, 0);
            continue;
        case X64_EXIT_INTERRUPT:
            switch (info.vector) {
            case 0x29: raise_guest(&s, cpu, STATUS_STACK_BUFFER_OVERRUN, (uint32_t)s.rip, eip, 1, (ULONG_PTR)(uint32_t)s.gpr[X64_RCX], 0); break;
            case 0x2c: raise_guest(&s, cpu, STATUS_ASSERTION_FAILURE, (uint32_t)s.rip, eip, 0, 0, 0); break;
            case 0x2d: raise_guest(&s, cpu, STATUS_BREAKPOINT, (uint32_t)s.rip, eip, 1, (ULONG_PTR)(uint32_t)s.gpr[X64_RAX], 0); break;
            case 4: raise_guest(&s, cpu, STATUS_INTEGER_OVERFLOW, (uint32_t)s.rip, eip, 0, 0, 0); break;
            default: raise_guest(&s, cpu, STATUS_ACCESS_VIOLATION, eip, eip, 2, 0, 0xffffffff); break;   // #GP on Windows
            }
            continue;
        case X64_EXIT_DIVIDE_ERROR:
            raise_guest(&s, cpu, STATUS_INTEGER_DIVIDE_BY_ZERO, eip, eip, 0, 0, 0);
            continue;
        case X64_EXIT_PRIVILEGED:
            raise_guest(&s, cpu, STATUS_PRIVILEGED_INSTRUCTION, eip, eip, 0, 0, 0);
            continue;
        case X64_EXIT_UNSUPPORTED:
            DbgPrint("wow64interp: unsupported instruction at %08x: %02x %02x %02x %02x %02x %02x\n", eip, info.bytes[0],
                     info.bytes[1], info.bytes[2], info.bytes[3], info.bytes[4], info.bytes[5]);
            raise_guest(&s, cpu, STATUS_ILLEGAL_INSTRUCTION, eip, eip, 0, 0, 0);
            continue;
        case X64_EXIT_ILLEGAL: case X64_EXIT_SYSCALL:
            raise_guest(&s, cpu, STATUS_ILLEGAL_INSTRUCTION, eip, eip, 0, 0, 0);
            continue;
        case X64_EXIT_STOP: case X64_EXIT_BUDGET:
            continue;
        }
    }
}

// A fault in 64-bit code. When it happened while interpreting 32-bit code, the
// 32-bit state is put back at the start of the faulting instruction and the
// record names that instruction; wow64's handler then passes it to the 32-bit
// side. (Registers the instruction had already changed stay changed.)
NTSTATUS WINAPI BTCpuResetToConsistentState(EXCEPTION_POINTERS *pointers)
{
    struct thread_slot *slot = thread_slot(FALSE);
    if (!slot || !slot->running || !slot->state) return STATUS_SUCCESS;
    X64State *s = slot->state;
    s->rip = s->insn_start;
    store_state(&cpu_area()->Context, s);
    pointers->ExceptionRecord->ExceptionAddress = host((uint32_t)s->insn_start);
    slot->running = 0;
    return STATUS_SUCCESS;
}

NTSTATUS WINAPI BTCpuProcessInit(void)
{
    ULONG_PTR base = 0;
    if (NtQueryInformationProcess(CURRENT_PROCESS, (PROCESSINFOCLASS)ProcessWineIosWowGuestBase, &base,
                                  sizeof base, NULL))
        base = 0;
    guest_base = base;

    UNICODE_STRING name;
    HMODULE ntdll;
    RtlInitUnicodeString(&name, L"ntdll.dll");
    if (!LdrGetDllHandle(NULL, 0, &name, &ntdll)) {
        void **dispatcher = RtlFindExportedRoutineByName(ntdll, "__wine_unix_call_dispatcher");
        if (dispatcher) unix_call_dispatcher = (unix_call_fn)*dispatcher;
    }

    // Two addresses below 4 GB that nothing runs (INT3 in case anything tries).
    void *page = NULL;
    SIZE_T size = 0x1000;
    NTSTATUS status = NtAllocateVirtualMemory(CURRENT_PROCESS, &page, 0x7fffffff, &size,
                                              MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (status) return status;
    memset(page, 0xcc, 0x1000);
    bop_syscall = (uint32_t)((uintptr_t)page - guest_base);
    bop_unix = bop_syscall + 16;

    WOW64INFO_ *wow64info = NtCurrentTeb()->TlsSlots[WOW64_TLS_WOW64INFO];
    if (wow64info) wow64info->CpuFlags |= WOW64_CPUFLAGS_SOFTWARE;
    return STATUS_SUCCESS;
}

void WINAPI BTCpuThreadInit(void)
{
    thread_slot(TRUE);
}

void *WINAPI BTCpuGetBopCode(void)
{
    return host(bop_syscall);
}

void *WINAPI __wine_get_unix_opcode(void)
{
    return host(bop_unix);
}

NTSTATUS WINAPI BTCpuGetContext(HANDLE thread, HANDLE process, void *unknown, WOW64_CONTEXT *context)
{
    (void)process; (void)unknown;
    return RtlWow64GetThreadContext(thread, context);
}

NTSTATUS WINAPI BTCpuSetContext(HANDLE thread, HANDLE process, void *unknown, WOW64_CONTEXT *context)
{
    (void)process; (void)unknown;
    return RtlWow64SetThreadContext(thread, context);
}

// The processor the interpreter is (x64_cpuid), not the host's.
BOOLEAN WINAPI BTCpuIsProcessorFeaturePresent(UINT feature)
{
    switch (feature) {
    case PF_COMPARE_EXCHANGE_DOUBLE: case PF_MMX_INSTRUCTIONS_AVAILABLE: case PF_XMMI_INSTRUCTIONS_AVAILABLE:
    case PF_RDTSC_INSTRUCTION_AVAILABLE: case PF_XMMI64_INSTRUCTIONS_AVAILABLE: case PF_SSE3_INSTRUCTIONS_AVAILABLE:
    case PF_NX_ENABLED: case PF_PAE_ENABLED: case PF_FASTFAIL_AVAILABLE:
        return TRUE;
    default:
        return FALSE;
    }
}

NTSTATUS WINAPI BTCpuTurboThunkControl(ULONG enable)
{
    return enable ? STATUS_NOT_SUPPORTED : STATUS_SUCCESS;
}

// Nothing is translated or cached, so there is nothing to flush or invalidate.
void WINAPI BTCpuFlushInstructionCache2(const void *address, SIZE_T size)
{
    (void)address; (void)size;
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, void *reserved)
{
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) LdrDisableThreadCalloutsForDll(instance);
    return TRUE;
}
