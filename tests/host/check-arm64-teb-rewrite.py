#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Madeira Converter Exception: see LICENSE-EXCEPTION.md
"""No JIT, step 4: build/signed-arm64/rewrite.py, which takes the x18 (TEB) reads and the
relocations out of the code of a plain ARM64 Windows module, so it can be signed.

With llvm-mc (the decoder these checks trust, not the rewriter's own), for each of the
modules a 32-bit process runs (ntdll, wow64, wow64win, wow64interp in
app/Madeira/aarch64-windows):
  - every instruction llvm-mc shows using x18 or w18, outside literal data, is now a
    branch to a trampoline, and no other instruction is;
  - each trampoline is the expected sequence, and its instruction is the original with
    x18 renamed to a register the original does not use; it branches back after the site;
  - each system call stub loads the same address PC-relative, its literal is zero and its
    relocation is gone: no relocation is left in code;
  - nothing else in the file changed;
  - done in two steps, as a signed wrapper would (the stubs on the file, then
    rewrite_image on the image mapped where the wrapper puts it), the result is the same.
Also, on random encodings of every instruction class the rewriter knows: what it must
refuse (x18 written, x18 beside SP) is refused.
With qemu-aarch64 and an aarch64-linux-gnu toolchain (or on ARM64 Linux):
  - every distinct x18 instruction in those modules, and the random encodings, run as the
    original (x18 = the TEB) and as rewritten (x18 garbage, the TEB found through
    TPIDR_EL0 and the slot) from the same registers, flags and memory: the same
    registers, flags and memory after, and x18 untouched (tests/x64-interp/teb-runner.c).
"""
from pathlib import Path
import importlib.util
import os
import platform
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[2]
farm = root / 'app/Madeira/aarch64-windows'
MODULES = ['ntdll.dll', 'wow64.dll', 'wow64win.dll', 'wow64interp.dll']
failures = 0

spec = importlib.util.spec_from_file_location('rewrite', root / 'build/signed-arm64/rewrite.py')
rewrite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rewrite)


def check(condition, label, detail=''):
    global failures
    print(('PASS: ' if condition else 'FAIL: ') + label)
    if not condition:
        failures += 1
        if detail:
            print(detail.rstrip())


mc = shutil.which('llvm-mc')


def disassemble(words):
    """llvm-mc's text for each word (whitespace collapsed), or None where it is not an instruction."""
    lines = '\n'.join(' '.join(f'0x{b:02x}' for b in struct.pack('<I', w)) for w in words)
    result = subprocess.run([mc, '--disassemble', '-triple=aarch64'], input=lines, capture_output=True, text=True)
    bad = {int(m[1]) for m in re.finditer(r'<stdin>:(\d+):\d+: warning: invalid instruction encoding', result.stderr)}
    text = [' '.join(line.split()) for line in result.stdout.splitlines()
            if line.strip() and not line.strip().startswith('.')]
    out, it = [], iter(text)
    for n in range(1, len(words) + 1):
        out.append(None if n in bad else next(it))
    return out


def memory(reg, offset):
    return f'[x{reg}, #{offset}]' if offset else f'[x{reg}]'


def regs_in(text):
    return {int(m[1]) for m in re.finditer(r'\b[xw](\d+)\b', text)}


X18 = re.compile(r'\b[xw]18\b')


def pe_sections(raw):
    nt = struct.unpack_from('<I', raw, 60)[0]
    count, optional_size = struct.unpack_from('<H', raw, nt + 6)[0], struct.unpack_from('<H', raw, nt + 20)[0]
    optional = nt + 24
    sections = []
    for i in range(count):
        off = optional + optional_size + 40 * i
        virtual, rva, length, position = struct.unpack_from('<IIII', raw, off + 8)
        sections.append({'name': raw[off:off + 8].rstrip(b'\0').decode(), 'rva': rva, 'length': length,
                         'position': position, 'flags': struct.unpack_from('<I', raw, off + 36)[0]})
    return {'base': struct.unpack_from('<Q', raw, optional + 24)[0], 'size': struct.unpack_from('<I', raw, optional + 56)[0],
            'reloc': struct.unpack_from('<II', raw, optional + 112 + 5 * 8), 'sections': sections}


def relocations(raw, pe):
    """(rva, type, file offset of the entry) for each base relocation."""
    rva, length = pe['reloc']
    out = []
    if not length:
        return out
    s = next(s for s in pe['sections'] if s['rva'] <= rva < s['rva'] + s['length'])
    off = s['position'] + rva - s['rva']
    end = off + length
    while off < end:
        page, block = struct.unpack_from('<II', raw, off)
        for e in range(off + 8, off + block, 2):
            v = struct.unpack_from('<H', raw, e)[0]
            out.append((page + (v & 4095), v >> 12, e))
        off += block
    return out


def literal_data(words):
    """Word indexes that an LDR (literal) loads: data in the code."""
    data = set()
    for i, insn in enumerate(words):
        size = {0x18: 4, 0x1C: 4, 0x98: 4, 0x58: 8, 0x5C: 8, 0xD8: 8, 0x9C: 16}.get(insn >> 24)
        if size:
            imm = (insn >> 5) & 0x7FFFF
            target = i + imm - ((imm & 0x40000) << 1)
            data.update(range(target, target + size // 4))
    return data


def branch_target(pc, insn):
    if insn & 0xFC000000 != 0x14000000:
        return None
    imm = insn & 0x3FFFFFF
    return pc + 4 * (imm - ((imm & 0x2000000) << 1))


def adrp_target(pc, insn):
    imm = ((insn >> 5) & 0x7FFFF) << 2 | (insn >> 29) & 3
    imm -= (imm & 0x100000) << 1
    return (pc & ~0xFFF) + (imm << 12)


def check_module(path):
    raw = path.read_bytes()
    pe = pe_sections(raw)
    slot = (pe['size'] + 0x3FFF) & ~0x3FFF
    tramp_rva = slot + 0x4000
    try:
        out, blob, report = rewrite.rewrite_pe(raw, tramp_rva, slot)
    except rewrite.Refused as error:
        check(False, f'{path.name}: rewritten', str(error))
        return []
    text = next(s for s in pe['sections'] if s['flags'] & 0x20000000)
    a, b = text['position'], text['position'] + text['length']
    old = list(struct.unpack_from(f"<{text['length'] // 4}I", raw, a))
    new = list(struct.unpack_from(f"<{text['length'] // 4}I", out, a))
    tramp = list(struct.unpack_from(f'<{len(blob) // 4}I', blob))
    relocs = relocations(raw, pe)
    in_code = [r for r in relocs if r[1] and text['rva'] <= r[0] < text['rva'] + text['length']]
    cleared = {e for _, _, e in in_code}

    # Nothing but the code and the cleared relocation entries changed.
    outside = [i for i in range(len(raw)) if raw[i] != out[i] and not a <= i < b and i - i % 2 not in cleared]
    entries_ok = all(struct.unpack_from('<H', out, e)[0] == 0 for e in cleared)
    left = [r for r in relocations(out, pe) if r[1] and text['rva'] <= r[0] < text['rva'] + text['length']]
    check(len(out) == len(raw) and not outside and entries_ok and not left,
          f'{path.name}: outside the code only relocation entries changed, and none is left in code '
          f'({len(cleared)} cleared)', f'changed bytes outside the code: {outside[:8]}; left: {left[:8]}')

    data = literal_data(old)
    old_text = disassemble(old)
    uses = {i for i, t in enumerate(old_text) if t and i not in data and X18.search(t)}
    rva = lambda i: text['rva'] + 4 * i
    sites = {i for i in range(len(old)) if new[i] != old[i] and i not in data
             and (branch_target(rva(i), new[i]) or -1) >= tramp_rva}
    check(sites == uses, f'{path.name}: each of the {len(uses)} instructions that use x18 is a branch to a trampoline, '
          'and nothing else is', f'not rewritten: {sorted(hex(rva(i)) for i in uses - sites)[:8]}; '
          f'rewritten without x18: {sorted(hex(rva(i)) for i in sites - uses)[:8]}')

    # The trampolines.
    tramp_text = disassemble(tramp)
    problems = []
    for i in sorted(sites):
        start = (branch_target(rva(i), new[i]) - tramp_rva) // 4
        seq, words = tramp_text[start:start + 10], tramp[start:start + 10]
        m = re.fullmatch(r'stp x(\d+), x(\d+), \[sp, #-16\]!', seq[0] or '')
        if not m:
            problems.append(f'{rva(i):#x}: {seq[0]}')
            continue
        ra, rb = int(m[1]), int(m[2])
        original = old_text[i]
        expected = [seq[0], f'mrs x{ra}, TPIDRRO_EL0', f'and x{ra}, x{ra}, #0xfffffffffffffff8', seq[3],
                    f'ldr w{rb}, {memory(rb, slot & 0xFFF)}', f'ldr x{ra}, [x{ra}, x{rb}]', f'ldr x{rb}, [sp, #8]',
                    re.sub(r'\b([xw])18\b', lambda g: f'{g[1]}{ra}', original), f'ldr x{ra}, [sp], #16', seq[9]]
        pc = tramp_rva + 4 * start
        if (seq != expected or {ra, rb} & regs_in(original) or re.search(r'\bw?sp\b', original)
                or not (seq[3] or '').startswith(f'adrp x{rb},') or adrp_target(pc + 12, words[3]) != slot & ~0xFFF
                or branch_target(pc + 36, words[9]) != rva(i) + 4):
            problems.append(f'{rva(i):#x}: {original} -> {seq}')
    check(not problems, f'{path.name}: each trampoline reads the TEB through TPIDRRO_EL0 and the slot, runs the '
          'instruction on it in a free register and branches back', '\n'.join(problems[:8]))

    # The system call stubs.
    stubs = []
    for target, _, _ in in_code:
        k = (target - text['rva']) // 4
        value = struct.unpack_from('<Q', raw, a + 4 * k)[0] - pe['base']
        loads = [j for j in range(k - 16, k) if old[j] >> 24 == 0x58 and j + ((old[j] >> 5) & 0x7FFFF) == k]
        ok = len(loads) == 1 and new[k] == new[k + 1] == 0
        if ok:
            j = loads[0]
            n = old[j] & 31
            ok = (old_text[j + 1] == f'ldr x{n}, [x{n}]' and (new_text := disassemble(new[j:j + 2]))
                  and (new_text[0] or '').startswith(f'adrp x{n},') and adrp_target(rva(j), new[j]) == value & ~0xFFF
                  and new_text[1] == f'ldr x{n}, {memory(n, value & 0xFFF)}')
        if not ok:
            stubs.append(f'{target:#x}')
    others = [i for i in range(len(old)) if old[i] != new[i] and i not in sites and i not in data
              and not any(i in (j, j + 1) for t, _, _ in in_code
                          for j in range((t - text['rva']) // 4 - 16, (t - text['rva']) // 4))]
    check(not stubs and not others and report['stubs'] == len(in_code),
          f"{path.name}: {len(in_code)} system call stubs load the dispatcher's address PC-relative",
          f'wrong: {stubs[:8]}; other changes: {[hex(rva(i)) for i in others[:8]]}')

    # The same through a wrapper (pull request #497's prepare.py): the stubs on the file,
    # then the x18 instructions in the image mapped by RVA at 16 KiB.
    stubs_only, none, _ = rewrite.rewrite_pe(raw)
    image = mapped(stubs_only, pe)
    wrapped, count = rewrite.rewrite_image(image, 0x4000, 0x4000 + tramp_rva, 0x4000 + slot)
    check(not none and wrapped == blob and count == report['teb_sites'] and image == mapped(out, pe),
          f'{path.name}: the same code and trampolines when a wrapper maps the image (rewrite_image)')
    return [old[i] for i in sorted(sites)]


def mapped(raw, pe):
    image = bytearray(pe['size'])
    headers = pe['sections'][0]['position'] if pe['sections'] else len(raw)
    image[:headers] = raw[:headers]
    for s in pe['sections']:
        image[s['rva']:s['rva'] + s['length']] = raw[s['position']:s['position'] + s['length']]
    return image


# --- Random encodings of each class ---------------------------------------------------

def random_cases(rng, count):
    """(word, refused) pairs: instructions that use x18, of every class rewrite.py knows."""
    r = lambda n: rng.randrange(n)
    reg = lambda: rng.choice([n for n in range(31) if n != 18])
    out = []
    while len(out) < count:
        kind = r(8)
        if kind == 0:                                # load/store, unsigned immediate
            size = r(4)
            opc = r(2) if size == 3 else r(3) if size == 2 else r(4)
            imm = r(0x7000 >> size)
            rt, rn = (reg(), 18) if r(3) else (18, reg())
            w = 0x39000000 | size << 30 | opc << 22 | imm << 10 | rn << 5 | rt
        elif kind == 1:                              # 9-bit immediate: unscaled, post, unprivileged, pre
            size = r(4)
            opc = r(2) if size == 3 else r(3) if size == 2 else r(4)
            rt, rn = (reg(), 18) if r(3) else (18, reg())
            w = 0x38000000 | size << 30 | opc << 22 | r(512) << 12 | r(4) << 10 | rn << 5 | rt
        elif kind == 2:                              # register offset
            size = r(4)
            opc = r(2) if size == 3 else r(3) if size == 2 else r(4)
            w = 0x38200800 | size << 30 | opc << 22 | rng.choice([2, 3, 6, 7]) << 13 | r(2) << 12 \
                | reg() << 16 | 18 << 5 | reg()
        elif kind == 3:                              # pair
            opc, load = rng.choice([(0, 0), (0, 1), (1, 1), (2, 0), (2, 1)])
            ts = [reg(), reg()]
            if not r(3):
                ts[r(2)], rn = 18, reg()
            else:
                rn = 18
            w = 0x28000000 | opc << 30 | r(4) << 23 | load << 22 | r(128) << 15 | ts[1] << 10 | rn << 5 | ts[0]
        elif kind in (4, 5):                         # logical or add/sub, shifted register
            sf = r(2)
            rn, rm = rng.choice([(18, reg()), (reg(), 18), (18, 18)])
            shift = r(4) if kind == 4 else r(3)
            base = 0x0A000000 | r(4) << 29 | r(2) << 21 if kind == 4 else 0x0B000000 | r(4) << 29
            w = base | sf << 31 | shift << 22 | rm << 16 | r(64 if sf else 32) << 10 | rn << 5 | rng.choice([reg(), 31])
        elif kind == 6:                              # add/sub, extended register
            rn, rm = rng.choice([(18, reg()), (reg(), 18)])
            w = 0x0B200000 | r(2) << 31 | r(4) << 29 | rm << 16 | r(8) << 13 | r(5) << 10 | 18 << 5 | reg() \
                if rn == 18 else 0x0B200000 | r(2) << 31 | r(4) << 29 | 18 << 16 | r(8) << 13 | r(5) << 10 | rn << 5 | reg()
        else:                                        # add/sub, immediate
            w = 0x11000000 | r(2) << 31 | r(4) << 29 | r(2) << 22 | r(4096) << 10 | 18 << 5 | rng.choice([reg(), 31])
        out.append(w)
    return out


REFUSED = {
    0xF9400252: 'ldr x18, [x18]',
    0xF8408E40: 'ldr x0, [x18, #8]!',
    0x8B3263E0: 'add x0, sp, x18',
    0xA90003F2: 'stp x18, x0, [sp]',
    0x8B010012: 'add x18, x0, x1',
    0x910003F2: 'mov x18, sp',
    0x91000240: None,                                # add x0, x18, #0: rewritten
    0xF100065F: None,                                # cmp x18, #1: rewritten (XZR, not SP)
}


def writes_x18_or_sp(text):
    """From llvm-mc's text: SP is an operand, or x18 is written (the destination of a load or
    of data processing, or a base written back)."""
    mnemonic, _, operands = text.partition(' ')
    first = operands.split(', ')
    return (bool(re.search(r'\b(w?sp)\b', operands)) or bool(re.search(r'\[x18(, #-?\d+)?\]!|\[x18\], #', operands))
            or (not mnemonic.startswith('st') and X18.fullmatch(first[0]) is not None)
            or (re.fullmatch(r'ldn?p(sw)?', mnemonic) is not None and X18.fullmatch(first[1]) is not None))


def main():
    if not mc:
        print('NOTE: no llvm-mc: the TEB rewrite was not checked')
        return
    # What must be refused is.
    wrong = []
    for word, name in REFUSED.items():
        try:
            rewrite.plan(word)
            refused = False
        except rewrite.Refused:
            refused = True
        if refused != (name is not None):
            wrong.append(f'{word:08x} {name or "allowed"}')
    texts = disassemble(list(REFUSED))
    wrong += [f'{w:08x} is {t}, not {n}' for (w, n), t in zip(REFUSED.items(), texts) if n and t != n]
    check(not wrong, 'instructions that write x18 or use SP beside it are refused', '\n'.join(wrong))

    words = []
    present = [farm / m for m in MODULES if (farm / m).is_file()]
    for path in present:
        words += check_module(path)
    if not present:
        print('NOTE: no modules in app/Madeira/aarch64-windows: only random encodings were checked')

    # Random encodings: the decodable ones that are not refused.
    rng = random.Random(0x7EB)
    candidates = random_cases(rng, 6000)
    texts = disassemble(candidates)
    synthetic, refused = [], 0
    for w, t in zip(candidates, texts):
        if not t or not X18.search(t):
            continue
        try:
            if rewrite.plan(w):
                synthetic.append(w)
        except rewrite.Refused:
            refused += 1
            if not writes_x18_or_sp(t):
                check(False, f'refused only for x18 written or SP: {t}')
    unused = sum(1 for w in synthetic if rewrite.plan(w) is None)
    check(len(synthetic) > 2000 and not unused,
          f'{len(synthetic)} random encodings that use x18 are rewritten ({refused} refused: x18 written or beside SP)')

    run_cases(sorted(set(words)), synthetic, rng)


# --- Running them ------------------------------------------------------------------------

def blob_common():
    """The prologue (loads x0-x30 and NZCV from x0) at 0x100 and the epilogue (stores them
    to the address at 0x300, then jumps to the address at 0x308) at 0x200."""
    pro = [0xF9407C01, 0xD51B4201]                   # ldr x1, [x0, #248]; msr nzcv, x1
    pro += [0xF9400000 | i << 10 | i for i in range(1, 31)]          # ldr xi, [x0, #8*i]
    pro += [0xF9400000, rewrite.b(0x100 + 4 * (len(pro) + 1), 0)]   # ldr x0, [x0]; b 0
    epi = [0xF81F0FE0]                               # str x0, [sp, #-16]!
    epi.append(0x58000000 | ((0x300 - 0x204) // 4) << 5)             # ldr x0, =out
    epi += [0xF9000000 | i << 10 | i for i in range(1, 31)]          # str xi, [x0, #8*i]
    epi += [0xF84107E1, 0xF9000001, 0xD53B4201, 0xF9007C01]          # ldr x1, [sp], #16; str x1, [x0]; mrs x1, nzcv; str
    epi.append(0x58000000 | ((0x308 - (0x200 + 4 * len(epi))) // 4) << 5 | 16)   # ldr x16, =done
    epi.append(0xD61F0200)                           # br x16
    blob = bytearray(0x200)
    struct.pack_into(f'<{len(pro)}I', blob, 0, *pro)
    struct.pack_into(f'<{len(epi)}I', blob, 0x100, *epi)
    return bytes(blob)


def case_input(word, rng):
    """The runner's input for one instruction, and its rewritten form."""
    blob, _ = rewrite.rewrite_code(bytearray(struct.pack('<II', word, rewrite.b(4, 0x200))), 0, 0x1000, 0x2000,
                                   tls='tpidr_el0')
    tramp = struct.unpack(f'<{len(blob) // 4}I', blob)
    rewritten = struct.pack('<I', rewrite.b(0, 0x1000)) + struct.pack('<I', rewrite.b(4, 0x200))
    gp, _, _ = rewrite.classify(word)
    regs = []
    for i in range(31):
        v = rng.getrandbits(64)
        if v >> 56 in (ord('S'), ord('T')):
            v ^= 1 << 63
        regs.append(v)
    memory = word & 0x0A000000 == 0x08000000         # loads and stores
    if memory:
        if gp['rn'] != 18:
            regs[gp['rn']] = ord('S') << 56 | 0x800000 + rng.randrange(-0x800, 0x800, 8)
        if 'rm' in gp and gp['rm'] != 18:
            regs[gp['rm']] = rng.randrange(0, 0x200)
    regs[18] = ord('T') << 56 | 0x800000
    regs.append(rng.randrange(16) << 28)             # NZCV
    data = struct.pack('<2I', word, rewrite.b(4, 0x200)) + rewritten + struct.pack(f'<I{len(tramp)}I', len(tramp), *tramp)
    data += struct.pack('<32Q', *regs) + struct.pack('<QQ', rng.randrange(0, 4096, 8), rng.getrandbits(64))
    return data


def run_cases(module_words, synthetic, rng):
    machine = platform.machine().lower()
    native = sys.platform == 'linux' and machine in ('aarch64', 'arm64')
    qemu, cross = shutil.which('qemu-aarch64'), shutil.which('aarch64-linux-gnu-gcc')
    cc = shutil.which('cc') if native else cross
    if not (native or (qemu and cross)) or not cc:
        print('NOTE: no ARM64 Linux, or qemu-aarch64 and aarch64-linux-gnu-gcc: the rewritten code was not run')
        return
    with tempfile.TemporaryDirectory() as folder:
        runner = Path(folder) / 'teb-runner'
        built = subprocess.run([cc, '-std=gnu11', '-O2', '-Wall', '-Wextra', '-Werror', '-static',
                                root / 'tests/x64-interp/teb-runner.c', '-o', runner], capture_output=True, text=True)
        check(built.returncode == 0, 'the ARM64 runner builds', built.stderr)
        if built.returncode:
            return
        for label, words in [('every distinct x18 instruction in the modules', module_words),
                             ('the random encodings', synthetic)]:
            if not words:
                continue
            data = struct.pack('<I', len(words)) + blob_common() + b''.join(case_input(w, rng) for w in words)
            result = subprocess.run(([] if native else [qemu]) + [runner], input=data, capture_output=True, timeout=900)
            lines = result.stdout.decode(errors='replace').splitlines()
            bad = [f'{w:08x} {t}: {line}' for w, t, line in zip(words, disassemble(words), lines) if line != 'ok']
            check(result.returncode == 0 and len(lines) == len(words) and not bad,
                  f'{label}: {len(words)} run the same rewritten, with x18 garbage and the TEB found through TPIDR_EL0 '
                  '(ARM64' + ('' if native else ', QEMU') + ')',
                  '\n'.join(bad[:12]) + result.stderr.decode(errors='replace')[-2000:]
                  + (f'\n{len(lines)} results for {len(words)} cases' if len(lines) != len(words) else ''))


if __name__ == '__main__':
    main()
    if failures:
        print(f'{failures} check(s) failed')
        sys.exit(1)
    print('All TEB rewrite checks passed.')
