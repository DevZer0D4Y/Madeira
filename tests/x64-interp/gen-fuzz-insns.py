#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Madeira Converter Exception: see LICENSE-EXCEPTION.md
"""Writes the instruction snippets fuzz.c runs natively and through X64Interpreter.

Each snippet is one instruction or a few (a branch and its two outcomes, an
equal-compare set-up for CMPXCHG). fuzz.c copies a snippet between a prologue
that loads random registers, flags, XMM registers and MXCSR and an epilogue
that stores them, runs it on the x86-64 host, then runs the same bytes at the
same address through the interpreter and compares everything.

Usage: gen-fuzz-insns.py OUT.S
"""
import sys

AF, CF, PF, ZF, SF, OF = 0x10, 0x1, 0x4, 0x40, 0x80, 0x800
ARITH = CF | PF | AF | ZF | SF | OF

# Template flags, matched in fuzz.c.
RCX_SMALL, R8_SMALL, BITOFF, POPF, XLAT, APPROX, NOMEM, SSE, RCX17 = (1 << i for i in range(9))
DIV = 1 << 9          # with the size in bits 16-19, signed in bit 20, a memory divisor in bit 21


def div(size, signed, mem=False):
    return DIV | (size << 16) | (signed << 20) | (mem << 21)


T = []


def t(asm, undef=0, flags=0):
    T.append((asm, undef, flags))


# --- integer arithmetic -----------------------------------------------------
for op in ['add', 'or', 'adc', 'sbb', 'and', 'sub', 'xor', 'cmp']:
    u = AF if op in ('and', 'or', 'xor') else 0
    for form in ['rax, rbx', 'ebx, ecx', 'dx, r9w', 'al, bh', 'sil, r10b', 'ah, cl', 'r13, r14',
                 'rax, 0x12345678', 'eax, -5', 'r11, -1', 'r10d, 0x7f', 'cx, 0x8001', 'bl, 0x80',
                 'al, 0x7f', 'ax, 3', 'rdx, -0x80000000',
                 'qword ptr [rsi+8], rbx', 'rbx, qword ptr [rsi+16]', 'dword ptr [rsi], 0x11',
                 'byte ptr [rdi+3], cl', 'word ptr [rsi+r8*2+4], 0x1234', 'ecx, dword ptr [rdi-12]']:
        t(f'{op} {form}', u, R8_SMALL)
    if op != 'cmp':
        for form in ['qword ptr [rsi+0x40], rax', 'dword ptr [rdi+4], 7', 'byte ptr [rsi+1], dl',
                     'word ptr [rsi+0x22], 0x7fff']:
            t(f'lock {op} {form}', u)

# Flag producers followed by flag consumers: the interpreter works flags out
# lazily, so each condition is read after each kind of producer.
for producer, u in [('cmp eax, ebx', 0), ('sub rcx, rdx', 0), ('add esi, edi', 0), ('cmp al, bl', 0), ('cmp r8w, r9w', 0),
                    ('and eax, ebx', AF), ('test rcx, rdx', AF), ('xor sil, dil', AF), ('or ax, bx', AF), ('inc eax', 0),
                    ('dec cl', 0), ('neg ebx', 0), ('cmp eax, dword ptr [rsi+8]', 0), ('sbb ecx, edx', 0),
                    ('adc rax, rbx', 0), ('sbb dl, 0x80', 0)]:
    for cc in ['o', 'no', 'b', 'ae', 'e', 'ne', 'be', 'a', 's', 'ns', 'p', 'np', 'l', 'ge', 'le', 'g']:
        t(f'{producer}; set{cc} r10b', u)
    t(f'{producer}; cmovle r11, r12; cmovb r13, r14; cmovo r15, rbx', u)
    t(f'{producer}; adc r10d, r11d', u)
    t(f'{producer}; jl 1f; mov r10d, 1; 1: jbe 2f; mov r11d, 2; 2: nop', u)
    t(f'{producer}; inc r12; setc r13b; sets r14b', u)
    t(f'{producer}; pushfq; pop r10', u)
    t(f'{producer}; shl r10d, 3', u | AF | OF)
    t(f'{producer}; shr r11, cl', u | AF | OF, RCX_SMALL)
    t(f'{producer}; rcl r12w, 1', u)
# Self-modifying code: each pass changes the immediate the next one runs (restored at the end).
t('lea rdx, [rip + 2f]; mov ecx, 3; 2: mov eax, 0x11111111; add ebx, eax; inc byte ptr [rdx+1]; dec ecx; jnz 2b; sub byte ptr [rdx+1], 3')
t('lea rdx, [rip + 2f]; mov byte ptr [rdx], 0x90; 2: int3; mov byte ptr [rdx], 0xcc')   # written before it first runs
t('mov ecx, 5; 1: add eax, ecx; dec ecx; jnz 1b')
t('xor eax, eax; 1: add eax, 3; cmp eax, 30; jl 1b; sete bl')
for op in ['inc', 'dec', 'neg', 'not']:
    for form in ['rax', 'ebx', 'cx', 'dl', 'ah', 'r12', 'r9b', 'qword ptr [rsi+8]', 'byte ptr [rdi]', 'word ptr [rsi+2]']:
        t(f'{op} {form}')
    t(f'lock {op} dword ptr [rsi+0x30]')

for form in ['rax, rbx', 'ecx, edx', 'si, di', 'al, ah', 'r9b, r10b', 'rax, 0x7fffffff', 'eax, 0x80000001',
             'bl, 0x81', 'qword ptr [rsi], rcx', 'byte ptr [rdi+1], 0x55', 'dword ptr [rsi+8], -1']:
    t(f'test {form}', AF)

# --- moves -------------------------------------------------------------------
for form in ['rax, rbx', 'eax, ebx', 'ax, bx', 'al, bl', 'ah, bl', 'bh, ch', 'sil, dil', 'r8, r15', 'r9d, r10d',
             'r11w, r12w', 'r13b, r14b', 'qword ptr [rsi+8], rax', 'rax, qword ptr [rsi+8]',
             'dword ptr [rdi], ecx', 'ecx, dword ptr [rdi]', 'word ptr [rsi+1], dx', 'dl, byte ptr [rsi+7]',
             'byte ptr [rdi], ah', 'qword ptr [rsi], -2', 'dword ptr [rsi], 0x12345678', 'word ptr [rsi], 0xbeef',
             'byte ptr [rsi], 0xfe', 'eax, 0xdeadbeef', 'rax, -1', 'r10, 0x7fffffff', 'cx, 0x1234', 'bh, 0x9a',
             'r15b, 0x11', 'rax, qword ptr [rsi+r8*8]', 'eax, dword ptr [rsi+r8*4-8]']:
    t(f'mov {form}', 0, R8_SMALL)
t('movabs rax, 0x1122334455667788')
t('movabs r14, 0x8000000000000001')
t('mov rax, qword ptr fs:[0]')
t('mov eax, dword ptr fs:[0x10]')
t('mov rax, qword ptr gs:[0x1008]')
t('mov dword ptr gs:[0x1010], ecx')
t('add rax, qword ptr gs:[rcx+0x800]', 0, RCX_SMALL)
for form in ['eax, bl', 'eax, bh', 'rax, cl', 'ecx, word ptr [rsi]', 'rdx, byte ptr [rsi+3]', 'ax, dl',
             'r9, r10w', 'esi, sil']:
    t(f'movzx {form}')
    t(f'movsx {form}')
t('movsxd rax, ebx')
t('movsxd r10, dword ptr [rsi+4]')
for form in ['rax, [rbx+rcx*4+0x10]', 'eax, [rbx+rcx]', 'ax, [rbx+rcx*2+0x1234]', 'r9, [rip+0x100]',
             'rdx, [rdx*8]', 'r12, [r13+r14*2-7]', 'eax, [ebx+ecx*2]', 'rcx, [rsp+0x18]', 'rax, [r12]',
             'rbp, [rbp+0x7f]', 'r11d, [r8+r9*8+0x80000000]']:
    t(f'lea {form}')
for form in ['rax, rbx', 'ecx, edx', 'ax, bx', 'al, ah', 'r8, r9', 'eax, eax', 'ebx, eax',
             'qword ptr [rsi+8], rdx', 'byte ptr [rdi], cl', 'dword ptr [rsi+4], r10d']:
    t(f'xchg {form}')
t('nop')
t('pause')
t('nop dword ptr [rax+rax*1+0x0]')
t('endbr64')
t('prefetcht0 byte ptr [rsi]')
t('push rax; pop rbx')
t('push r12; pop r13')
t('push 0x12345678; pop rax')
t('push -3; pop rcx')
t('push qword ptr [rsi+8]; pop qword ptr [rdi]')
t('push rsp; pop rax')
t('push rax; pop rsp')
t('push rbx; push rcx; pop qword ptr [rsp]; pop rdx')
t('push word ptr [rsi]; pop ax; .byte 0x66, 0x68, 0x34, 0x12; pop bx')   # push word 0x1234
for i in ['cbw', 'cwde', 'cdqe', 'cwd', 'cdq', 'cqo']:
    t(i)
t('enter 0x20, 0; mov qword ptr [rbp-8], rax; leave')
t('push rbp; mov rbp, rsp; sub rsp, 0x40; mov qword ptr [rsp], rbx; leave')
t('xlatb', 0, XLAT)
t('mov rax, qword ptr [rsi]; bswap rax')
t('bswap ecx')
t('bswap r9')
t('bswap r12d')

# --- multiply and divide -----------------------------------------------------
for form in ['rbx', 'ecx', 'dx', 'bl', 'ah', 'qword ptr [rsi]', 'byte ptr [rdi]']:
    t(f'mul {form}', SF | ZF | AF | PF)
    t(f'imul {form}', SF | ZF | AF | PF)
for form in ['rax, rbx', 'ecx, edx', 'si, di', 'r9, qword ptr [rsi]', 'rax, rbx, 0x1234', 'eax, ecx, -3',
             'dx, bx, 0x7fff', 'r10, qword ptr [rsi+8], -0x80000000', 'ebx, dword ptr [rdi], 5']:
    t(f'imul {form}', SF | ZF | AF | PF)
for size, reg in [(8, 'rbx'), (4, 'ebx'), (2, 'bx'), (1, 'bl')]:
    t(f'div {reg}', ARITH, div(size, 0))
    t(f'idiv {reg}', ARITH, div(size, 1))
for size, ptr in [(8, 'qword'), (4, 'dword'), (2, 'word'), (1, 'byte')]:
    t(f'div {ptr} ptr [rsi]', ARITH, div(size, 0, True))
    t(f'idiv {ptr} ptr [rsi]', ARITH, div(size, 1, True))

# --- shifts and rotates --------------------------------------------------------
for op in ['shl', 'shr', 'sar', 'rol', 'ror', 'rcl', 'rcr']:
    rot = op in ('rol', 'ror', 'rcl', 'rcr')
    for reg, bits in [('rax', 64), ('ebx', 32), ('dx', 16), ('al', 8), ('ah', 8), ('r10d', 32), ('qword ptr [rsi]', 64),
                      ('word ptr [rdi+2]', 16)]:
        one = AF if not rot else 0
        many = (OF if rot else AF | OF)
        cl = many | (CF if bits < 32 and op in ('shl', 'shr') else 0)
        t(f'{op} {reg}, 1', one)
        t(f'{op} {reg}, 7', many)
        t(f'{op} {reg}, {bits - 1}', many)
        t(f'{op} {reg}, cl', cl)
for op in ['shld', 'shrd']:
    for dst, src, bits in [('rax', 'rbx', 64), ('ecx', 'edx', 32), ('si', 'di', 16), ('qword ptr [rsi]', 'r9', 64)]:
        t(f'{op} {dst}, {src}, 1', AF)
        t(f'{op} {dst}, {src}, 13', AF | OF)
        t(f'{op} {dst}, {src}, cl', AF | OF, RCX17 if bits == 16 else 0)

# --- bits ------------------------------------------------------------------------
for op in ['bt', 'bts', 'btr', 'btc']:
    for form in ['rax, rbx', 'ecx, edx', 'si, di', 'rax, 63', 'ebx, 33', 'cx, 17']:
        t(f'{op} {form}', OF | SF | AF | PF)
    t(f'{op} qword ptr [rsi+0x100], rbx', OF | SF | AF | PF, BITOFF)
    t(f'{op} dword ptr [rsi+0x100], ebx', OF | SF | AF | PF, BITOFF)
    t(f'{op} word ptr [rsi+0x100], bx', OF | SF | AF | PF, BITOFF)
    t(f'{op} dword ptr [rsi+4], 9', OF | SF | AF | PF)
    if op != 'bt':
        t(f'lock {op} qword ptr [rsi+0x100], rbx', OF | SF | AF | PF, BITOFF)
for op in ['bsf', 'bsr']:
    for form in ['rax, rbx', 'ecx, edx', 'si, di', 'r9, qword ptr [rsi]']:
        t(f'{op} {form}', CF | OF | SF | AF | PF)
        t(f'xor ebx, ebx; xor edx, edx; xor edi, edi; mov qword ptr [rsi], 0; {op} {form}', CF | OF | SF | AF | PF)
for op in ['tzcnt', 'lzcnt']:
    for form in ['rax, rbx', 'ecx, edx', 'si, di', 'r9, qword ptr [rsi]']:
        t(f'{op} {form}', OF | SF | AF | PF)
        t(f'xor ebx, ebx; xor edx, edx; {op} {form}', OF | SF | AF | PF)
for form in ['rax, rbx', 'ecx, edx', 'si, di', 'r9, qword ptr [rsi]']:
    t(f'popcnt {form}')

# --- conditions -------------------------------------------------------------------
for cc in ['o', 'no', 'b', 'ae', 'e', 'ne', 'be', 'a', 's', 'ns', 'p', 'np', 'l', 'ge', 'le', 'g']:
    t(f'set{cc} al')
    t(f'set{cc} byte ptr [rdi+5]')
    t(f'cmov{cc} rax, rbx')
    t(f'cmov{cc} ecx, edx')
    t(f'cmov{cc} r9w, word ptr [rsi]')
    t(f'cmov{cc} eax, dword ptr [rsi]')
    t(f'j{cc} {{L}}; or r14, 1; {{L}}: nop')
    t(f'j{cc} {{L}}; .fill 200, 1, 0x90; or r14, 2; {{L}}: nop')
t('jrcxz {L}; or r14, 1; {L}: nop', 0, RCX_SMALL)
t('loop {L}; or r14, 1; {L}: nop', 0, RCX_SMALL)
t('loope {L}; or r14, 1; {L}: nop', 0, RCX_SMALL)
t('loopne {L}; or r14, 1; {L}: nop', 0, RCX_SMALL)
t('call {L}; or r14, 1; {L}: pop rcx')
t('lea rax, [rip+{L}]; push rax; ret; or r14, 1; {L}: nop')
t('lea rax, [rip+{L}]; push rbx; push rax; ret 8; or r14, 1; {L}: nop')
t('lea rax, [rip+{L}]; jmp rax; or r14, 1; {L}: nop')
t('lea rax, [rip+{L}]; mov qword ptr [rsi], rax; jmp qword ptr [rsi]; or r14, 1; {L}: nop')
t('lea rax, [rip+{L}]; call rax; {L}: pop rdx')
t('jmp {L}; or r14, 1; {L}: nop')

# --- flags --------------------------------------------------------------------------
for i in ['stc', 'clc', 'cmc', 'std', 'cld', 'lahf', 'sahf', 'pushfq; pop rax', 'mov ah, bl; sahf']:
    t(i)
t('push rbx; popfq', 0, POPF)
t('push rbx; popfq; pushfq; pop rax', 0, POPF)

# --- atomics -------------------------------------------------------------------------
for size, reg, acc, ptr in [(8, 'rbx', 'rax', 'qword'), (4, 'ebx', 'eax', 'dword'), (2, 'bx', 'ax', 'word'), (1, 'bl', 'al', 'byte')]:
    t(f'cmpxchg {ptr} ptr [rsi+0x20], {reg}')
    t(f'lock cmpxchg {ptr} ptr [rsi+0x20], {reg}')
    t(f'mov {acc}, {ptr} ptr [rsi+0x20]; lock cmpxchg {ptr} ptr [rsi+0x20], {reg}')
    t(f'xadd {ptr} ptr [rsi+0x28], {reg}')
    t(f'lock xadd {ptr} ptr [rsi+0x28], {reg}')
for dst, src, acc in [('rcx', 'rbx', 'rax'), ('ecx', 'ebx', 'eax'), ('cx', 'bx', 'ax'), ('cl', 'bl', 'al')]:
    t(f'cmpxchg {dst}, {src}')
    t(f'mov {acc}, {dst}; cmpxchg {dst}, {src}')
    t(f'xadd {dst}, {src}')
t('xadd eax, eax')
t('cmpxchg16b xmmword ptr [rsi+0x100]')
t('mov rax, qword ptr [rsi+0x100]; mov rdx, qword ptr [rsi+0x108]; lock cmpxchg16b xmmword ptr [rsi+0x100]')
t('cmpxchg8b qword ptr [rsi+0x100]')
t('mov eax, dword ptr [rsi+0x100]; mov edx, dword ptr [rsi+0x104]; lock cmpxchg8b qword ptr [rsi+0x100]')
t('mfence')
t('lfence')
t('sfence')

# --- strings ----------------------------------------------------------------------------
for i in ['movsb', 'movsw', 'movsd', 'movsq', 'stosb', 'stosw', 'stosd', 'stosq', 'lodsb', 'lodsw', 'lodsd', 'lodsq',
          'scasb', 'scasd', 'scasq', 'cmpsb', 'cmpsw', 'cmpsq']:
    t(i, 0, RCX_SMALL)
    if i.startswith(('scas', 'cmps')):
        t('repe ' + i, 0, RCX_SMALL)
        t('repne ' + i, 0, RCX_SMALL)
        t(f'mov rdx, rdi; xchg rsi, rdx; mov r9, rcx; shl r9, 3; cld; repe {i}', 0, RCX_SMALL)   # equal ranges
    else:
        t('rep ' + i, 0, RCX_SMALL)
t('lea rdi, [rsi+5]; cld; rep movsb', 0, RCX_SMALL)       # overlapping, forward
t('lea rdi, [rsi+3]; cld; rep movsq', 0, RCX_SMALL)
t('lea rsi, [rdi+3]; cld; rep movsb', 0, RCX_SMALL)
t('cld; rep movsb', 0, RCX_SMALL)
t('cld; rep stosb', 0, RCX_SMALL)
t('mov al, byte ptr [rdi+9]; cld; repne scasb', 0, RCX_SMALL)

# --- x87 control -------------------------------------------------------------------------
t('fnstcw word ptr [rdi]')
t('mov word ptr [rdi], 0x27f; fldcw word ptr [rdi]; fnstcw word ptr [rdi+2]')
t('fninit; fnstcw word ptr [rdi]; fnstsw ax')
t('fnstsw word ptr [rdi]')
t('fwait')

# --- SSE ---------------------------------------------------------------------------------------
S = lambda asm, undef=0, flags=0: t(asm, undef, flags | SSE)
# SSE after a flag-setting instruction (COMISD sets EFLAGS), and with a segment base.
for producer in ['cmp eax, ebx', 'add rcx, rdx', 'test esi, edi']:
    S(f'{producer}; comisd xmm0, xmm1', AF if producer.startswith('test') else 0)
    S(f'{producer}; ucomiss xmm2, xmm3; setb r10b', AF if producer.startswith('test') else 0)
S('movss xmm1, dword ptr gs:[0x1010]')
S('addsd xmm2, qword ptr gs:[0x1020]')
S('movdqu xmmword ptr gs:[0x1030], xmm3')
S('pshufd xmm4, xmmword ptr gs:[0x1040], 0x1b')
for i in ['movaps xmm0, xmm1', 'movaps xmm2, xmmword ptr [rsi+0x20]', 'movaps xmmword ptr [rdi+0x30], xmm3',
          'movups xmm4, xmmword ptr [rsi+3]', 'movups xmmword ptr [rdi+5], xmm5', 'movapd xmm9, xmm10',
          'movupd xmm11, xmmword ptr [rsi+1]', 'movss xmm1, xmm2', 'movss xmm1, dword ptr [rsi+4]',
          'movss dword ptr [rdi+8], xmm4', 'movsd xmm1, xmm2', 'movsd xmm3, qword ptr [rsi+8]',
          'movsd qword ptr [rdi+8], xmm4', 'movlps xmm0, qword ptr [rsi]', 'movhps xmm0, qword ptr [rsi]',
          'movlpd xmm1, qword ptr [rsi+8]', 'movhpd xmm1, qword ptr [rsi+8]', 'movlps qword ptr [rdi], xmm2',
          'movhps qword ptr [rdi], xmm2', 'movlpd qword ptr [rdi+8], xmm3', 'movhpd qword ptr [rdi+8], xmm3',
          'movhlps xmm4, xmm5', 'movlhps xmm4, xmm5', 'movd xmm0, eax', 'movd xmm0, dword ptr [rsi]',
          'movq xmm1, rax', 'movq rax, xmm1', 'movd ecx, xmm2', 'movd dword ptr [rdi], xmm2', 'movq xmm3, xmm4',
          'movq xmm3, qword ptr [rsi]', 'movq qword ptr [rdi], xmm2', 'movdqa xmm6, xmm7',
          'movdqa xmm6, xmmword ptr [rsi+0x40]', 'movdqa xmmword ptr [rdi+0x40], xmm8', 'movdqu xmm12, xmmword ptr [rsi+7]',
          'movdqu xmmword ptr [rdi+9], xmm13', 'lddqu xmm14, xmmword ptr [rsi+1]', 'movddup xmm0, xmm1',
          'movddup xmm0, qword ptr [rsi]', 'movsldup xmm2, xmm3', 'movshdup xmm2, xmmword ptr [rsi]',
          'movntps xmmword ptr [rdi+0x60], xmm1', 'movntdq xmmword ptr [rdi+0x60], xmm1', 'movnti dword ptr [rdi], eax',
          'movnti qword ptr [rdi], r9', 'maskmovdqu xmm1, xmm2', 'movmskps eax, xmm1', 'movmskpd ecx, xmm2',
          'pmovmskb edx, xmm3', 'pextrw eax, xmm1, 3', 'pinsrw xmm1, eax, 5', 'pinsrw xmm1, word ptr [rsi], 2']:
    S(i)
for op in ['add', 'sub', 'mul', 'div', 'min', 'max', 'sqrt']:
    for kind in ['ps', 'pd', 'ss', 'sd']:
        mem = {'ps': 'xmmword', 'pd': 'xmmword', 'ss': 'dword', 'sd': 'qword'}[kind]
        S(f'{op}{kind} xmm0, xmm1')
        S(f'{op}{kind} xmm9, {mem} ptr [rsi+0x10]')
for i in ['rcpps xmm0, xmm1', 'rcpss xmm0, xmm1', 'rsqrtps xmm2, xmm3', 'rsqrtss xmm2, xmm3']:
    S(i, 0, APPROX)
for op in ['and', 'andn', 'or', 'xor']:
    S(f'{op}ps xmm0, xmm1')
    S(f'{op}pd xmm2, xmmword ptr [rsi]')
for kind in ['ps', 'pd', 'ss', 'sd']:
    for pred in range(8):
        S(f'cmp{kind} xmm0, xmm1, {pred}')
    mem = {'ps': 'xmmword', 'pd': 'xmmword', 'ss': 'dword', 'sd': 'qword'}[kind]
    S(f'cmp{kind} xmm3, {mem} ptr [rsi], 1')
for i in ['comiss xmm0, xmm1', 'ucomiss xmm0, xmm1', 'comisd xmm2, xmm3', 'ucomisd xmm2, qword ptr [rsi]',
          'ucomiss xmm4, dword ptr [rsi]']:
    S(i)
for i in ['cvtsi2ss xmm0, eax', 'cvtsi2ss xmm0, rax', 'cvtsi2sd xmm1, ecx', 'cvtsi2sd xmm1, rcx',
          'cvtsi2sd xmm1, qword ptr [rsi]', 'cvtsi2ss xmm2, dword ptr [rsi]', 'cvtss2si eax, xmm1', 'cvtss2si rax, xmm1',
          'cvttss2si ecx, xmm2', 'cvttss2si rcx, dword ptr [rsi]', 'cvtsd2si eax, xmm3', 'cvtsd2si rax, xmm3',
          'cvttsd2si edx, xmm4', 'cvttsd2si rdx, qword ptr [rsi]', 'cvtss2sd xmm5, xmm6', 'cvtss2sd xmm5, dword ptr [rsi]',
          'cvtsd2ss xmm7, xmm8', 'cvtsd2ss xmm7, qword ptr [rsi]', 'cvtps2pd xmm0, xmm1', 'cvtps2pd xmm0, qword ptr [rsi]',
          'cvtpd2ps xmm2, xmm3', 'cvtdq2ps xmm4, xmm5', 'cvtps2dq xmm6, xmm7', 'cvttps2dq xmm8, xmm9',
          'cvtdq2pd xmm10, xmm11', 'cvtpd2dq xmm12, xmm13', 'cvttpd2dq xmm14, xmm15']:
    S(i)
for i in ['shufps xmm0, xmm1, 0x1b', 'shufps xmm0, xmmword ptr [rsi], 0xe4', 'shufpd xmm2, xmm3, 1', 'shufpd xmm2, xmm3, 2',
          'pshufd xmm0, xmm1, 0x1b', 'pshufd xmm0, xmmword ptr [rsi], 0x4e', 'pshufhw xmm2, xmm3, 0x93',
          'pshuflw xmm4, xmm5, 0x39', 'unpcklps xmm0, xmm1', 'unpckhps xmm0, xmm1', 'unpcklpd xmm2, xmm3',
          'unpckhpd xmm2, xmmword ptr [rsi]', 'pshufb xmm0, xmm1', 'pshufb xmm2, xmmword ptr [rsi]',
          'palignr xmm0, xmm1, 5', 'palignr xmm0, xmm1, 16', 'palignr xmm0, xmm1, 20', 'palignr xmm0, xmm1, 35']:
    S(i)
for op in ['paddb', 'paddw', 'paddd', 'paddq', 'psubb', 'psubw', 'psubd', 'psubq', 'paddsb', 'paddsw', 'paddusb',
           'paddusw', 'psubsb', 'psubsw', 'psubusb', 'psubusw', 'pmullw', 'pmulhw', 'pmulhuw', 'pmuludq', 'pmaddwd',
           'psadbw', 'pavgb', 'pavgw', 'pminub', 'pmaxub', 'pminsw', 'pmaxsw', 'pand', 'pandn', 'por', 'pxor',
           'pcmpeqb', 'pcmpeqw', 'pcmpeqd', 'pcmpgtb', 'pcmpgtw', 'pcmpgtd', 'packsswb', 'packssdw', 'packuswb',
           'punpcklbw', 'punpcklwd', 'punpckldq', 'punpcklqdq', 'punpckhbw', 'punpckhwd', 'punpckhdq', 'punpckhqdq',
           'psllw', 'pslld', 'psllq', 'psrlw', 'psrld', 'psrlq', 'psraw', 'psrad']:
    S(f'{op} xmm0, xmm1')
    S(f'{op} xmm10, xmmword ptr [rsi+0x30]')
for op in ['psllw', 'pslld', 'psllq', 'psrlw', 'psrld', 'psrlq', 'psraw', 'psrad']:
    for n in [0, 1, 7, 15, 31, 40, 63, 64]:
        S(f'{op} xmm3, {n}')
    S(f'mov eax, ecx; and eax, 0x7f; movd xmm1, eax; {op} xmm2, xmm1')
for op in ['pslldq', 'psrldq']:
    for n in [0, 1, 5, 15, 16, 200]:
        S(f'{op} xmm4, {n}')
for i in ['haddps xmm0, xmm1', 'haddpd xmm0, xmm1', 'hsubps xmm2, xmm3', 'hsubpd xmm2, xmm3', 'addsubps xmm4, xmm5',
          'addsubpd xmm4, xmmword ptr [rsi]']:
    S(i)
S('stmxcsr dword ptr [rdi]')
# FXSAVE's x87 part holds whatever this process last left there: compare MXCSR, the
# control word and the XMM registers through registers, not the memory.
S('fxsave [rsi+0x200]; mov eax, dword ptr [rsi+0x218]; movzx ebx, word ptr [rsi+0x200]; movdqu xmm9, xmmword ptr [rsi+0x2d0]', 0, NOMEM)
S('fxsave [rsi+0x200]; mov dword ptr [rsi+0x218], 0x7fc0; movdqu xmmword ptr [rsi+0x2a0], xmm7; fxrstor [rsi+0x200]; stmxcsr dword ptr [rdi]', 0, NOMEM)
S('stmxcsr dword ptr [rdi]; mov eax, dword ptr [rdi]; xor eax, 0x6000; mov dword ptr [rdi+4], eax; ldmxcsr dword ptr [rdi+4]; cvtsd2si rax, xmm1; addss xmm2, xmm3')
S('stmxcsr dword ptr [rdi]; mov eax, dword ptr [rdi]; or eax, 0x8040; mov dword ptr [rdi+4], eax; ldmxcsr dword ptr [rdi+4]; mulps xmm2, xmm3; addsd xmm4, xmm5')

ANCHOR = 0x30100000
STATE_IN, STATE_OUT = ANCHOR, ANCHOR + 0x400
SAVE_RSP, SCRATCH = ANCHOR + 0x800, ANCHOR + 0xf00

pre = ['push rbx', 'push rbp', 'push r12', 'push r13', 'push r14', 'push r15',
       f'mov qword ptr [{SAVE_RSP:#x}], rsp', f'ldmxcsr dword ptr [{STATE_IN + 0x88:#x}]']
pre += [f'movdqu xmm{i}, xmmword ptr [{STATE_IN + 0x100 + 16 * i:#x}]' for i in range(16)]
pre += [f'mov rsp, {SCRATCH:#x}', f'push qword ptr [{STATE_IN + 0x80:#x}]', 'popfq']
regs = ['rax', 'rcx', 'rdx', 'rbx', 'rsp', 'rbp', 'rsi', 'rdi'] + [f'r{i}' for i in range(8, 16)]
pre += [f'mov {r}, qword ptr [{STATE_IN + 8 * i:#x}]' for i, r in enumerate(regs) if r != 'rsp']
pre += [f'mov rsp, qword ptr [{STATE_IN + 0x20:#x}]']
post = [f'mov qword ptr [{STATE_OUT + 8 * i:#x}], {r}' for i, r in enumerate(regs)]
post += [f'mov rsp, {SCRATCH:#x}', 'pushfq', f'pop qword ptr [{STATE_OUT + 0x80:#x}]', 'cld']
post += [f'movdqu xmmword ptr [{STATE_OUT + 0x100 + 16 * i:#x}], xmm{i}' for i in range(16)]
post += [f'stmxcsr dword ptr [{STATE_OUT + 0x88:#x}]', f'mov rsp, qword ptr [{SAVE_RSP:#x}]',
         'pop r15', 'pop r14', 'pop r13', 'pop r12', 'pop rbp', 'pop rbx', 'ret']

out = ['.intel_syntax noprefix', '.section .rodata.fuzz_code,"a"', '.globl stub_pre_start, stub_pre_end, stub_post_start, stub_post_end',
       'stub_pre_start:'] + ['    ' + i for i in pre] + ['stub_pre_end:', 'stub_post_start:'] + ['    ' + i for i in post] + ['stub_post_end:']
names = []
for n, (asm, undef, flags) in enumerate(T):
    body = asm.replace('{L}', f'.Lt{n}')
    out += [f't{n}_start:'] + ['    ' + part.strip() for part in body.split(';')] + [f't{n}_end:']
    names.append(asm)
out += ['.section .rodata.fuzz_names,"a"']
for n, asm in enumerate(names):
    escaped = asm.replace('\\', '\\\\').replace('"', '\\"')
    out += [f'n{n}: .asciz "{escaped}"']
out += ['.section .data.rel.ro.fuzz,"aw"', '.globl fuzz_templates, fuzz_template_count', '.balign 8', 'fuzz_templates:']
for n, (asm, undef, flags) in enumerate(T):
    out += [f'    .quad t{n}_start, t{n}_end, n{n}, {undef:#x}, {flags:#x}']
out += ['fuzz_template_count:', f'    .quad {len(T)}', '.section .note.GNU-stack,"",@progbits']
open(sys.argv[1], 'w').write('\n'.join(out) + '\n')
print(f'{len(T)} templates')
