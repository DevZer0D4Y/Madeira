#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Madeira Converter Exception: see LICENSE-EXCEPTION.md
"""Writes the 32-bit snippets fuzz32.c runs natively and fuzz.c replays through
X64Interpreter's 32-bit (WoW64) mode: integer code as old compilers emit it,
x87 and MMX, which is what older games are made of.

Usage: gen-fuzz-insns32.py OUT.S
"""
import sys

AF, CF, PF, ZF, SF, OF = 0x10, 0x1, 0x4, 0x40, 0x80, 0x800
ARITH = CF | PF | AF | ZF | SF | OF

# Template flags, matched in fuzz-common.h.
(RCX_SMALL, R8_SMALL, BITOFF, POPF, XLAT, APPROX, NOMEM, SSE, RCX17, DIV) = (1 << i for i in range(10))
X87, MMX, X87RC, FSWAX, CMPC1, X87INT = (1 << i for i in range(10, 16))
X87NEAR, X87UNIT, X87FULL = 1 << 22, 1 << 23, 1 << 24


def div(size, signed, mem=False):
    return DIV | (size << 16) | (signed << 20) | (mem << 21)


T = []


def t(asm, undef=0, flags=0):
    T.append((asm, undef, flags))


# --- integer code --------------------------------------------------------------
for op in ['add', 'or', 'adc', 'sbb', 'and', 'sub', 'xor', 'cmp']:
    u = AF if op in ('and', 'or', 'xor') else 0
    for form in ['eax, ebx', 'ecx, edx', 'si, di', 'al, bh', 'ah, cl', 'eax, 0x12345678', 'ebx, -5', 'cx, 0x8001',
                 'dl, 0x80', 'dword ptr [esi+8], ebx', 'ebx, dword ptr [esi+16]', 'byte ptr [edi+3], cl',
                 'word ptr [esi+2], 0x1234', 'dword ptr [0x30201010], 7']:
        t(f'{op} {form}', u)
    if op != 'cmp':
        t(f'lock {op} dword ptr [esi+0x40], eax', u)
for op in ['inc', 'dec']:
    for form in ['eax', 'ebx', 'esp', 'edi', 'ax', 'cx', 'dl', 'ah', 'dword ptr [esi]', 'word ptr [edi+2]']:
        t(f'{op} {form}')
    t(f'lock {op} dword ptr [esi+0x30]')
for op in ['neg', 'not']:
    for form in ['eax', 'cx', 'bh', 'dword ptr [esi+4]']:
        t(f'{op} {form}')
for form in ['eax, ebx', 'cx, dx', 'al, 0x81', 'dword ptr [esi], 0x7fffffff']:
    t(f'test {form}', AF)
for form in ['eax, ebx', 'ax, bx', 'al, ah', 'bh, ch', 'dword ptr [esi+8], eax', 'eax, dword ptr [esi+8]',
             'word ptr [edi], cx', 'dl, byte ptr [esi+7]', 'dword ptr [esi], 0x12345678', 'eax, 0xdeadbeef',
             'eax, dword ptr [0x30201020]', 'dword ptr [0x30202000], ecx', 'eax, dword ptr [esi+ecx*4]']:
    t(f'mov {form}', 0, RCX_SMALL)
for form in ['eax, bl', 'eax, bh', 'ecx, word ptr [esi]', 'ax, dl']:
    t(f'movzx {form}')
    t(f'movsx {form}')
for form in ['eax, [ebx+ecx*4+0x10]', 'eax, [ebx+ecx]', 'ax, [ebx+ecx*2+0x1234]', 'edx, [edx*8]', 'ecx, [esp+0x18]',
             'ebp, [ebp-0x7f]', 'eax, [0x12345678]']:
    t(f'lea {form}')
for form in ['eax, ebx', 'ecx, edx', 'ax, bx', 'al, ah', 'dword ptr [esi+8], edx']:
    t(f'xchg {form}')
for i in ['push eax; pop ebx', 'push 0x12345678; pop eax', 'push -3; pop ecx', 'push dword ptr [esi+8]; pop dword ptr [edi]',
          'push esp; pop eax', 'push eax; pop esp', 'push ebx; push ecx; pop dword ptr [esp]; pop edx',
          'push word ptr [esi]; pop ax', 'pushad; popad', 'pushad; pop eax; pop ebx; pop ecx; pop edx; pop esi; pop edi; pop ebp; pop ebp',
          'pushaw; popaw', 'push ds; pop eax', 'push es; push cs; push ss; pop ebx; pop ecx; pop edx', 'push ds; pop es',
          'push fs; pop eax', 'push gs; pop ebx', 'cbw', 'cwde', 'cwd', 'cdq', 'nop', 'pause',
          'enter 0x20, 0; mov dword ptr [ebp-8], eax; leave', 'push ebp; mov ebp, esp; sub esp, 0x40; leave',
          'xlatb', 'bswap eax', 'bswap esi']:
    t(i, 0, XLAT if 'xlat' in i else 0)
for op, u in [('daa', OF), ('das', OF), ('aaa', OF | SF | ZF | PF), ('aas', OF | SF | ZF | PF), ('aam', OF | AF | CF),
              ('aam 16', OF | AF | CF), ('aad', OF | AF | CF), ('aad 7', OF | AF | CF), ('.byte 0xd6', 0)]:
    t(op, u)
t('xor eax, eax; into')
for form in ['ebx', 'cx', 'bl', 'dword ptr [esi]']:
    t(f'mul {form}', SF | ZF | AF | PF)
    t(f'imul {form}', SF | ZF | AF | PF)
for form in ['eax, ebx', 'cx, dx', 'eax, ebx, 0x1234', 'ecx, dword ptr [esi], -3']:
    t(f'imul {form}', SF | ZF | AF | PF)
for size, reg in [(4, 'ebx'), (2, 'bx'), (1, 'bl')]:
    t(f'div {reg}', ARITH, div(size, 0))
    t(f'idiv {reg}', ARITH, div(size, 1))
t('div dword ptr [esi]', ARITH, div(4, 0, True))
t('idiv dword ptr [esi]', ARITH, div(4, 1, True))
for op in ['shl', 'shr', 'sar', 'rol', 'ror', 'rcl', 'rcr']:
    rot = op in ('rol', 'ror', 'rcl', 'rcr')
    for reg, bits in [('eax', 32), ('dx', 16), ('al', 8), ('dword ptr [esi]', 32)]:
        t(f'{op} {reg}, 1', 0 if rot else AF)
        t(f'{op} {reg}, 7', OF if rot else AF | OF)
        t(f'{op} {reg}, cl', (OF if rot else AF | OF) | (CF if bits < 32 and op in ('shl', 'shr') else 0))
for op in ['shld', 'shrd']:
    t(f'{op} eax, ebx, 13', AF | OF)
    t(f'{op} ecx, edx, cl', AF | OF)
    t(f'{op} si, di, cl', AF | OF, RCX17)
for op in ['bt', 'bts', 'btr', 'btc']:
    t(f'{op} eax, ebx', OF | SF | AF | PF)
    t(f'{op} eax, 17', OF | SF | AF | PF)
    t(f'{op} dword ptr [esi+0x100], ebx', OF | SF | AF | PF, BITOFF)
for op in ['bsf', 'bsr']:
    t(f'{op} eax, ebx', CF | OF | SF | AF | PF)
    t(f'xor ebx, ebx; {op} eax, ebx', CF | OF | SF | AF | PF)
for cc in ['o', 'b', 'e', 'be', 's', 'p', 'l', 'le', 'ae', 'ne', 'a', 'g']:
    t(f'set{cc} al')
    t(f'cmov{cc} eax, ebx')
    t(f'j{cc} {{L}}; or edi, 1; {{L}}: nop')
for i in ['jecxz {L}; or edi, 1; {L}: nop', 'loop {L}; or edi, 1; {L}: nop', 'loopne {L}; or edi, 1; {L}: nop',
          'call {L}; or edi, 1; {L}: pop ecx',
          # Targets found from EIP: a snippet runs from a copy, not where it was assembled.
        # (.byte 0x05 is ADD EAX, imm32: clang's assembler takes no label difference there.)
          'call {L}x; {L}x: pop eax; .byte 0x05; .long {L} - {L}x; push eax; ret; or edi, 1; {L}: nop',
          'call {L}x; {L}x: pop eax; .byte 0x05; .long {L} - {L}x; push ebx; push eax; ret 4; {L}: nop',
          'call {L}x; {L}x: pop eax; .byte 0x05; .long {L} - {L}x; jmp eax; or edi, 1; {L}: nop',
          'call {L}x; {L}x: pop eax; .byte 0x05; .long {L} - {L}x; call eax; {L}: pop edx',
          'call {L}x; {L}x: pop eax; .byte 0x05; .long {L} - {L}x; mov dword ptr [esi], eax; call dword ptr [esi]; {L}: pop edx']:
    t(i, 0, RCX_SMALL)
for i in ['stc', 'clc', 'cmc', 'std', 'cld', 'lahf', 'sahf', 'pushfd; pop eax']:
    t(i)
t('push ebx; popfd', 0, POPF)
for i in ['movsb', 'movsd', 'stosd', 'lodsw', 'scasb', 'cmpsd']:
    t(i, 0, RCX_SMALL)
    t(('repe ' if i[:4] in ('scas', 'cmps') else 'rep ') + i, 0, RCX_SMALL)
t('lea edi, [esi+5]; cld; rep movsb', 0, RCX_SMALL)
t('cld; rep movsd', 0, RCX_SMALL)
t('cmpxchg dword ptr [esi+0x20], ebx')
t('mov eax, dword ptr [esi+0x20]; lock cmpxchg dword ptr [esi+0x20], ebx')
t('xadd dword ptr [esi+0x28], ebx')
t('cmpxchg8b qword ptr [esi+0x100]')
t('mov eax, dword ptr [esi+0x100]; mov edx, dword ptr [esi+0x104]; lock cmpxchg8b qword ptr [esi+0x100]')

# --- x87 -----------------------------------------------------------------------------
X = lambda asm, undef=0, flags=0: t(asm, undef, flags | X87)
for size in ['dword', 'qword']:
    X(f'fld {size} ptr [esi+8]')
    X(f'fst {size} ptr [edi]')
    X(f'fstp {size} ptr [edi+8]')
    for op in ['fadd', 'fmul', 'fsub', 'fsubr', 'fdiv', 'fdivr']:
        X(f'{op} {size} ptr [esi+16]')
    X(f'fcom {size} ptr [esi+16]; fnstsw ax', 0, FSWAX)
    X(f'fcomp {size} ptr [esi+16]; fnstsw ax', 0, FSWAX)
X('fld tbyte ptr [esi+32]; fstp tbyte ptr [edi+16]', 0, X87INT)
X('fstp tbyte ptr [edi+16]')
X('fld tbyte ptr [esi+32]', 0, X87INT)
for i in range(4):
    X(f'fld st({i})')
    X(f'fst st({i})')
    X(f'fstp st({i})')
    X(f'fxch st({i})')
    X(f'ffree st({i})')
for op in ['fadd', 'fmul', 'fsub', 'fsubr', 'fdiv', 'fdivr']:
    X(f'{op} st, st(1)')
    X(f'{op} st(2), st')
    X(f'{op}p st(1), st')
    X(f'{op}p st(2), st')
for size in ['word', 'dword']:
    for op in ['fiadd', 'fimul', 'fisub', 'fisubr', 'fidiv', 'fidivr']:
        X(f'{op} {size} ptr [esi+24]', 0, X87INT)
    X(f'ficom {size} ptr [esi+24]; fnstsw ax', 0, FSWAX | X87INT)
    X(f'ficomp {size} ptr [esi+24]; fnstsw ax', 0, FSWAX | X87INT)
for size in ['word', 'dword', 'qword']:
    X(f'fild {size} ptr [esi+24]', 0, X87INT)
    X(f'fist {size} ptr [edi]', 0, X87RC) if size != 'qword' else None
    X(f'fistp {size} ptr [edi]', 0, X87RC)
    X(f'fisttp {size} ptr [edi]')
    X(f'fild {size} ptr [esi+24]; fistp {size} ptr [edi]', 0, X87INT)
X('fild dword ptr [esi+24]; fbstp tbyte ptr [edi]; fbld tbyte ptr [edi]', 0, X87INT)
X('fbstp tbyte ptr [edi]', 0, X87RC)
for c in ['fld1', 'fldz', 'fldpi', 'fldl2e', 'fldl2t', 'fldlg2', 'fldln2']:
    X(c, 0, X87RC)
for i in ['fcom st(1)', 'fcomp st(2)', 'fcompp', 'fucom st(1)', 'fucomp st(3)', 'fucompp', 'ftst', 'fxam']:
    X(f'{i}; fnstsw ax', 0, FSWAX | (CMPC1 if i == 'fxam' else 0))
X('fcom st(1); fnstsw ax; sahf; seta cl; setb dl', 0, FSWAX)
X('fcomp st(1); fnstsw ax; test ah, 0x41; setz bl', 0, FSWAX)
for i in ['fcomi st, st(1)', 'fcomip st, st(2)', 'fucomi st, st(1)', 'fucomip st, st(3)']:
    X(i)
for cc in ['b', 'e', 'be', 'u', 'nb', 'ne', 'nbe', 'nu']:
    X(f'fcmov{cc} st, st(2)')
for i in ['fchs', 'fabs', 'fsqrt', 'frndint', 'fscale', 'fxtract', 'fincstp', 'fdecstp', 'fninit', 'fnclex', 'fnop', 'fwait']:
    X(i, 0, X87RC if i in ('frndint', 'fsqrt') else 0)
# FPREM reports a partial remainder (C2) when the exponents are far apart; these stay close.
for i in ['fprem', 'fprem1', 'fprem; fnstsw ax', 'fprem1; fnstsw ax']:
    X(i, 0, X87NEAR | (FSWAX if 'fnstsw' in i else 0))
for i in ['fsin', 'fcos', 'fsincos', 'fptan', 'fpatan', 'fyl2x']:
    X(i, 0, APPROX | X87NEAR)
for i in ['f2xm1', 'fyl2xp1']:
    X(i, 0, APPROX | X87UNIT)
X('fnstcw word ptr [edi]')
X('mov word ptr [edi], 0x0e7f; fldcw word ptr [edi]; fld1; fldpi; fdivp st(1), st; fistp dword ptr [edi+4]')
X('fnstsw ax', 0, FSWAX)
X('fnstsw word ptr [edi]; mov ax, word ptr [edi]', 0, FSWAX | NOMEM)
X('fnstenv [edi]; mov eax, dword ptr [edi]; mov ebx, dword ptr [edi+4]; mov ecx, dword ptr [edi+8]', 0, NOMEM | FSWAX)
X('fnstenv [edi]; fldenv [edi]', 0, NOMEM)
X('fnsave [edi]; frstor [edi]', 0, NOMEM)
X('fnsave [edi]; fld1; frstor [edi]', 0, NOMEM)
X('fld st(0); fmul st, st(1); fadd st(2), st; fst qword ptr [edi]; fistp dword ptr [edi+8]', 0, X87NEAR)
# Arithmetic on any 80-bit values at 24-, 53- and 64-bit precision, in every rounding direction.
F = lambda asm: X(asm, 0, X87FULL)
for op in ['fadd', 'fmul', 'fsub', 'fsubr', 'fdiv', 'fdivr']:
    F(f'{op} st, st(1)')
    F(f'{op} st(1), st')
    F(f'{op}p st(1), st')
    for size in ['dword', 'qword']:
        F(f'{op} {size} ptr [esi+16]')
    F(f'fi{op[1:]} dword ptr [esi+24]')
for i in ['fsqrt', 'fscale', 'frndint', 'fxtract', 'fst dword ptr [edi]', 'fst qword ptr [edi]', 'fstp tbyte ptr [edi]',
          'fist dword ptr [edi]', 'fistp qword ptr [edi]', 'fld st(0); fmul st, st(1); fadd st(2), st; fdiv st, st(3); fsqrt; fsub st, st(1)']:
    F(i)
F('fld st(0); fsub st, st(1)')                      # exact zeros: their sign follows the rounding control
F('fld st(0); fchs; fadd st, st(1)')
X('fcom st(1); fnstsw ax', 0, X87FULL | FSWAX)
X('fxam; fnstsw ax', 0, X87FULL | FSWAX | CMPC1)

# --- MMX -------------------------------------------------------------------------------
M = lambda asm, undef=0, flags=0: t(asm, undef, flags | MMX)
for op in ['paddb', 'paddw', 'paddd', 'paddq', 'psubb', 'psubw', 'psubd', 'psubq', 'paddsb', 'paddsw', 'paddusb',
           'paddusw', 'psubsb', 'psubsw', 'psubusb', 'psubusw', 'pmullw', 'pmulhw', 'pmulhuw', 'pmuludq', 'pmaddwd',
           'psadbw', 'pavgb', 'pavgw', 'pminub', 'pmaxub', 'pminsw', 'pmaxsw', 'pand', 'pandn', 'por', 'pxor',
           'pcmpeqb', 'pcmpeqw', 'pcmpeqd', 'pcmpgtb', 'pcmpgtw', 'pcmpgtd', 'packsswb', 'packssdw', 'packuswb',
           'punpcklbw', 'punpcklwd', 'punpckldq', 'punpckhbw', 'punpckhwd', 'punpckhdq',
           'psllw', 'pslld', 'psllq', 'psrlw', 'psrld', 'psrlq', 'psraw', 'psrad']:
    M(f'{op} mm0, mm1')
    M(f'{op} mm5, {"dword" if op.startswith("punpckl") else "qword"} ptr [esi+0x30]')
for op in ['psllw', 'pslld', 'psllq', 'psrlw', 'psrld', 'psrlq', 'psraw', 'psrad']:
    for n in [0, 1, 7, 15, 31, 33, 63, 64]:
        M(f'{op} mm3, {n}')
for i in ['movd mm0, eax', 'movd mm1, dword ptr [esi]', 'movd eax, mm2', 'movd dword ptr [edi], mm3', 'movq mm4, mm5',
          'movq mm6, qword ptr [esi+8]', 'movq qword ptr [edi+8], mm7', 'pshufw mm0, mm1, 0x1b', 'pshufw mm2, qword ptr [esi], 0x4e',
          'pinsrw mm0, eax, 2', 'pinsrw mm1, word ptr [esi], 3', 'pextrw eax, mm1, 3', 'pmovmskb eax, mm2', 'maskmovq mm1, mm2',
          'movntq qword ptr [edi], mm3', 'emms', 'paddd mm0, mm1; emms']:
    M(i)
S = lambda asm, undef=0, flags=0: t(asm, undef, flags | SSE | MMX)
for i in ['cvtpi2ps xmm0, mm1', 'cvtpi2ps xmm2, qword ptr [esi]', 'cvtps2pi mm0, xmm1', 'cvttps2pi mm0, xmm1',
          'cvtpi2pd xmm0, mm1', 'cvtpd2pi mm0, xmm1', 'cvttpd2pi mm2, xmm3', 'movq2dq xmm0, mm1', 'movdq2q mm0, xmm1',
          'addps xmm0, xmm1', 'mulsd xmm2, qword ptr [esi]', 'cvttsd2si eax, xmm3', 'movdqu xmm4, xmmword ptr [esi+3]',
          'pshufd xmm5, xmm6, 0x1b', 'ucomiss xmm0, xmm1', 'paddw xmm7, xmm0']:
    S(i)

ANCHOR = 0x30100000
IN, OUT = ANCHOR, ANCHOR + 0x400
SAVE_ESP, SCRATCH = ANCHOR + 0x800, ANCHOR + 0xf00
regs = ['eax', 'ecx', 'edx', 'ebx', 'esp', 'ebp', 'esi', 'edi']
pre = ['push ebx', 'push ebp', 'push esi', 'push edi', f'mov dword ptr [{SAVE_ESP:#x}], esp',
       f'fxrstor [{IN + 0x40:#x}]', f'mov esp, {SCRATCH:#x}', f'push dword ptr [{IN + 0x20:#x}]', 'popfd']
pre += [f'mov {r}, dword ptr [{IN + 4 * i:#x}]' for i, r in enumerate(regs) if r != 'esp']
pre += [f'mov esp, dword ptr [{IN + 0x10:#x}]']
post = [f'mov dword ptr [{OUT + 4 * i:#x}], {r}' for i, r in enumerate(regs)]
post += [f'mov esp, {SCRATCH:#x}', 'pushfd', f'pop dword ptr [{OUT + 0x20:#x}]', 'cld', f'fxsave [{OUT + 0x40:#x}]',
         f'mov esp, dword ptr [{SAVE_ESP:#x}]', 'pop edi', 'pop esi', 'pop ebp', 'pop ebx', 'ret']

out = ['.intel_syntax noprefix', '.code32', '.section .rodata.fuzz_code,"a"',
       '.globl stub_pre_start, stub_pre_end, stub_post_start, stub_post_end', 'stub_pre_start:']
out += ['    ' + i for i in pre] + ['stub_pre_end:', 'stub_post_start:'] + ['    ' + i for i in post] + ['stub_post_end:']
for n, (asm, undef, flags) in enumerate(T):
    body = asm.replace('{L}', f'.Lt{n}')
    out += [f't{n}_start:'] + ['    ' + part.strip() for part in body.split(';')] + [f't{n}_end:']
out += ['.section .rodata.fuzz_names,"a"']
for n, (asm, undef, flags) in enumerate(T):
    out += [f'n{n}: .asciz "{asm}"']
out += ['.section .data.rel.ro.fuzz,"aw"', '.globl fuzz_templates, fuzz_template_count', '.balign 4', 'fuzz_templates:']
for n, (asm, undef, flags) in enumerate(T):
    out += [f'    .long t{n}_start, t{n}_end, n{n}', f'    .quad {undef:#x}, {flags:#x}']
out += ['fuzz_template_count:', f'    .long {len(T)}', '.section .note.GNU-stack,"",@progbits']
open(sys.argv[1], 'w').write('\n'.join(out) + '\n')
print(f'{len(T)} templates')
