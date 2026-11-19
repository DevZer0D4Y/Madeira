#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Madeira Converter Exception: see LICENSE-EXCEPTION.md
"""Prepare a plain ARM64 Windows PE to run as signed code on iOS (docs/NO-JIT.md, step 4).

The 32-bit side of a WoW64 process runs four native ARM64 modules (Wine's aarch64
ntdll, wow64 and wow64win, and the CPU module). To run them as signed code instead of
from the JIT pool, their code must not change after signing, and must not need the two
things Madeira's loader does to it today when it copies it into the JIT pool:

- Reading the TEB through x18. iOS does not keep x18; Madeira keeps the TEB in a
  pthread TSD slot, at an offset found at run time (ios_teb_tls_slot_offset), and its
  loader rewrites every x18 instruction to read it there (ios_jit_patch_x18). This does
  the same before signing: each instruction that uses x18 becomes a branch to a
  trampoline that reads the TEB from the TSD slot and runs the instruction with that
  register instead. The offset is not an immediate but a 32-bit data slot the loader
  fills (as upstream pull request #497 does for ARM64EC). Plain ARM64 has no reserved
  scratch registers (ARM64EC has x13 and x14), so the trampoline saves two on the
  stack, which plain ARM64 keeps 16-byte aligned:

      stp  xA, xB, [sp, #-16]!
      mrs  xA, TPIDRRO_EL0
      and  xA, xA, #~7               the low 3 bits are the CPU number
      adrp xB, slot
      ldr  wB, [xB, :lo12:slot]      the TEB's offset in the TSD
      ldr  xA, [xA, xB]              the TEB
      ldr  xB, [sp, #8]
      <the instruction, x18 -> xA>
      ldr  xA, [sp], #16
      b    <the next instruction>

  A and B are registers the instruction does not use. Flags are not touched.

- Relocations inside code. Wine's ARM64 system call stubs load the address of
  __wine_syscall_dispatcher from a literal after the stub:
      ldr x16, <literal> ; ldr x16, [x16] ; blr x16 ; ret ; <literal: 8 bytes, DIR64>
  The first two become `adrp x16, dispatcher ; ldr x16, [x16, :lo12:dispatcher]`, the
  literal becomes zeros and its relocation entry is cleared.

Anything this does not know how to rewrite is refused, with the instruction: an
instruction that writes x18 or writes back to it, x18 next to SP, a relocation in code
that is not a system call stub's literal.

Two ways to use it:
- With build/signed-pe/prepare.py (pull request #497): first this, on the file, for the
  system call stubs only (the result has no relocation in code, which prepare.py
  requires); then, where prepare.py calls thread_state.rewrite for ARM64EC, call
      rewrite_image(image, PE_OFFSET, tramp_start, thread_offset + 4)
  for plain ARM64: the trampolines go where ARM64EC's do, the slot is the same (offset
  4 of madeira_pe_thread_state), and it returns the trampolines and how many sites.
      rewrite.py IN.dll OUT.dll
- On its own, everything on the file, the trampolines at RVA `--trampolines` (after the
  image, within branch reach) and the slot at `--slot`, both for a wrapper to place:
      rewrite.py IN.dll OUT.dll --trampolines RVA --slot RVA --blob TRAMPOLINES.bin
                 [--tls tpidrro_el0|tpidr_el0] [--report REPORT.json]

tpidr_el0 is for tests on Linux, where TPIDRRO_EL0 cannot be set.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

SCRATCH = [16, 17, 15, 14, 13, 12, 11, 10, 9]
MRS = {'tpidrro_el0': 0xD53BD060, 'tpidr_el0': 0xD53BD040}


class Refused(ValueError):
    pass


# --- Instruction classes -------------------------------------------------------------
# Each returns (gpr_fields, writes, sp_fields) for an instruction that has register
# fields, or None: gpr_fields maps a field name to its value for every field that names
# a general-purpose register; writes is the set of those fields the instruction writes
# (a load's Rt, a data-processing Rd, a base with writeback); sp_fields are the fields
# where 31 means SP.

FIELDS = {'rt': 0, 'rn': 5, 'rt2': 10, 'rm': 16, 'rd': 0}


def field(insn, name):
    return (insn >> FIELDS[name]) & 31


def prefetch(insn):
    """PRFM and PRFUM: size 11, opc 10; their Rt field is the prefetch operation."""
    return insn >> 30 == 3 and (insn >> 22) & 3 == 2


def classify(insn):
    gp, writes, sp = {}, set(), set()
    if insn & 0x3B000000 == 0x39000000:                      # load/store, unsigned immediate
        vector, load = (insn >> 26) & 1, (insn >> 22) & 3 != 0
        if not vector and not prefetch(insn):
            gp['rt'] = field(insn, 'rt')
            if load:
                writes.add('rt')
        gp['rn'] = field(insn, 'rn')
        sp.add('rn')
        return gp, writes, sp
    if insn & 0x3B200000 == 0x38000000:                      # load/store, 9-bit immediate
        vector, load, mode = (insn >> 26) & 1, (insn >> 22) & 3 != 0, (insn >> 10) & 3
        if not vector and not prefetch(insn):
            gp['rt'] = field(insn, 'rt')
            if load:
                writes.add('rt')
        gp['rn'] = field(insn, 'rn')
        sp.add('rn')
        if mode in (1, 3):
            writes.add('rn')                                 # pre- or post-index writeback
        return gp, writes, sp
    if insn & 0x3B200C00 == 0x38200800:                      # load/store, register offset
        vector, load = (insn >> 26) & 1, (insn >> 22) & 3 != 0
        if not vector and not prefetch(insn):
            gp['rt'] = field(insn, 'rt')
            if load:
                writes.add('rt')
        gp['rn'] = field(insn, 'rn')
        gp['rm'] = field(insn, 'rm')
        sp.add('rn')
        return gp, writes, sp
    if insn & 0x3A000000 == 0x28000000:                      # load/store pair
        vector, load, mode = (insn >> 26) & 1, (insn >> 22) & 1, (insn >> 23) & 3
        if not vector:
            gp['rt'] = field(insn, 'rt')
            gp['rt2'] = field(insn, 'rt2')
            if load:
                writes |= {'rt', 'rt2'}
        gp['rn'] = field(insn, 'rn')
        sp.add('rn')
        if mode in (1, 3):
            writes.add('rn')
        return gp, writes, sp
    if insn & 0x1F000000 == 0x0A000000:                      # logical, shifted register (MOV is ORR)
        return {'rd': field(insn, 'rd'), 'rn': field(insn, 'rn'), 'rm': field(insn, 'rm')}, {'rd'}, set()
    if insn & 0x1F200000 == 0x0B000000:                      # add/sub, shifted register
        return {'rd': field(insn, 'rd'), 'rn': field(insn, 'rn'), 'rm': field(insn, 'rm')}, {'rd'}, set()
    # In the two below, register 31 is SP, except as the destination of ADDS and SUBS
    # (CMN and CMP), where it is XZR.
    sp = {'rn'} if insn & (1 << 29) else {'rd', 'rn'}
    if insn & 0x1F200000 == 0x0B200000:                      # add/sub, extended register
        return {'rd': field(insn, 'rd'), 'rn': field(insn, 'rn'), 'rm': field(insn, 'rm')}, {'rd'}, sp
    if insn & 0x1F800000 == 0x11000000:                      # add/sub, immediate
        return {'rd': field(insn, 'rd'), 'rn': field(insn, 'rn')}, {'rd'}, sp
    return None


def plan(insn):
    """None if the instruction does not use x18; else (new instruction with x18 -> A, A, B).
    Raises Refused for what cannot be rewritten."""
    c = classify(insn)
    if c is None:
        return None
    gp, writes, sp = c
    hits = [f for f, r in gp.items() if r == 18]
    if not hits:
        return None
    if any(f in writes for f in hits):
        raise Refused('writes x18')
    if any(f in sp and r == 31 for f, r in gp.items()):
        raise Refused('uses SP beside x18')
    used = set(gp.values())
    free = [r for r in SCRATCH if r not in used]
    a, b = free[0], free[1]
    new = insn
    for f in hits:
        new = (new & ~(31 << FIELDS[f])) | (a << FIELDS[f])
    return new, a, b


# --- Encodings --------------------------------------------------------------------------

def b(frm, to):
    delta = to - frm
    if delta % 4 or not -(1 << 27) <= delta < (1 << 27):
        raise Refused(f'branch from {frm:#x} to {to:#x} out of reach')
    return 0x14000000 | ((delta >> 2) & 0x3FFFFFF)


def adrp(reg, pc, target):
    delta = (target >> 12) - (pc >> 12)
    if not -(1 << 20) <= delta < (1 << 20):
        raise Refused('ADRP out of reach')
    v = delta & 0x1FFFFF
    return 0x90000000 | (v & 3) << 29 | (v >> 2) << 5 | reg


def trampoline(new, a, b_, pc, back, slot, tls):
    words = [
        0xA9BF0000 | b_ << 10 | 31 << 5 | a,                 # stp xA, xB, [sp, #-16]!
        MRS[tls] | a,                                        # mrs xA, <tls>
        0x927DF000 | a << 5 | a,                             # and xA, xA, #~7
        adrp(b_, pc + 12, slot),                             # adrp xB, slot
        0xB9400000 | ((slot & 0xFFF) // 4) << 10 | b_ << 5 | b_,   # ldr wB, [xB, :lo12:slot]
        0xF8606800 | b_ << 16 | a << 5 | a,                  # ldr xA, [xA, xB]
        0xF94007E0 | b_,                                     # ldr xB, [sp, #8]
        new,                                                 # the instruction, x18 -> xA
        0xF84107E0 | a,                                      # ldr xA, [sp], #16
    ]
    words.append(b(pc + 4 * len(words), back))
    return words


# --- The text ------------------------------------------------------------------------

def literal_words(text):
    """Byte offsets of the words that are data: what LDR (literal) instructions load."""
    data = set()
    for i in range(0, len(text) - 3, 4):
        insn = struct.unpack_from('<I', text, i)[0]
        size = {0x18: 4, 0x1C: 4, 0x98: 4, 0x58: 8, 0x5C: 8, 0xD8: 8, 0x9C: 16}.get(insn >> 24)
        if not size:
            continue
        imm19 = ((insn >> 5) & 0x7FFFF)
        imm19 -= (imm19 & 0x40000) << 1
        target = i + imm19 * 4
        if 0 <= target < len(text):
            for w in range(target, min(target + size, len(text)), 4):
                data.add(w)
    return data


def rewrite_code(text, address, trampolines, slot, tls='tpidrro_el0'):
    """Sends every instruction in `text` (a bytearray of code at `address`) that uses x18
    through a trampoline, the trampolines to be placed at `trampolines` and the TSD offset
    at `slot`: three addresses in one space, RVAs or a wrapper's own. Changes `text` in
    place; returns the trampolines (bytes) and how many instructions."""
    data = literal_words(text)
    words, sites = [], 0
    for i in range(0, len(text) - 3, 4):
        if i in data:
            continue
        insn = struct.unpack_from('<I', text, i)[0]
        try:
            p = plan(insn)
        except Refused as error:
            raise Refused(f'{error}: {insn:08x} at {address + i:#x}')
        if p is None:
            continue
        new, a, b_ = p
        pc = trampolines + 4 * len(words)
        words += trampoline(new, a, b_, pc, address + i + 4, slot, tls)
        struct.pack_into('<I', text, i, b(address + i, pc))
        sites += 1
    return struct.pack(f'<{len(words)}I', *words), sites


def rewrite_stub(text, address, literal, target):
    """The system call stub that loads the 8-byte literal at `literal` (`ldr xN, <literal>;
    ldr xN, [xN]`) loads `target` PC-relative instead; the literal becomes zeros."""
    off = literal - address
    found = None
    for i in range(max(0, off - 64), off, 4):
        insn = struct.unpack_from('<I', text, i)[0]
        if insn & 0xFF000000 != 0x58000000:
            continue
        imm19 = (insn >> 5) & 0x7FFFF
        if i + (imm19 - ((imm19 & 0x40000) << 1)) * 4 == off:
            found = i
    if found is None:
        raise Refused(f'relocation in code at {literal:#x} is not a literal an LDR loads')
    n = struct.unpack_from('<I', text, found)[0] & 31
    if struct.unpack_from('<I', text, found + 4)[0] != (0xF9400000 | n << 5 | n) or target % 8:
        raise Refused(f'literal at {literal:#x} is not loaded as `ldr x{n}, <literal>; ldr x{n}, [x{n}]`')
    struct.pack_into('<II', text, found, adrp(n, address + found, target),
                     0xF9400000 | ((target & 0xFFF) // 8) << 10 | n << 5 | n)
    struct.pack_into('<Q', text, off, 0)


# --- The PE ---------------------------------------------------------------------------

def headers(raw):
    nt = struct.unpack_from('<I', raw, 60)[0]
    if raw[nt:nt + 4] != b'PE\0\0' or struct.unpack_from('<H', raw, nt + 4)[0] != 0xAA64:
        raise Refused('not a plain ARM64 PE')
    count, optional_size = struct.unpack_from('<H', raw, nt + 6)[0], struct.unpack_from('<H', raw, nt + 20)[0]
    optional = nt + 24
    sections = []
    for i in range(count):
        off = optional + optional_size + 40 * i
        virtual, rva, length, position = struct.unpack_from('<IIII', raw, off + 8)
        sections.append({'rva': rva, 'length': length, 'position': position,
                         'code': bool(struct.unpack_from('<I', raw, off + 36)[0] & 0x20000000)})
    return {'optional': optional, 'base': struct.unpack_from('<Q', raw, optional + 24)[0],
            'size': struct.unpack_from('<I', raw, optional + 56)[0], 'sections': sections}


def rewrite_pe(raw, trampolines=None, slot=None, tls='tpidrro_el0'):
    """The system call stubs, and with `trampolines` and `slot` (RVAs after the image)
    the x18 instructions. Returns the PE, the trampolines and a report."""
    raw = bytearray(raw)
    pe = headers(raw)
    code = [s for s in pe['sections'] if s['code']]
    if trampolines is not None and (min(trampolines, slot) < pe['size'] or slot % 4 or trampolines % 4):
        raise Refused('the trampolines and the slot go after the image, aligned')

    def file_offset(rva):
        for s in pe['sections']:
            if s['rva'] <= rva < s['rva'] + s['length']:
                return s['position'] + rva - s['rva']
        raise Refused(f'rva {rva:#x} is not in the file')

    in_code = lambda rva: next((s for s in code if s['rva'] <= rva < s['rva'] + s['length']), None)
    reloc_rva, reloc_len = struct.unpack_from('<II', raw, pe['optional'] + 112 + 5 * 8)
    off = file_offset(reloc_rva) if reloc_len else 0
    end, stubs = off + reloc_len, []
    while off < end:
        page, block = struct.unpack_from('<II', raw, off)
        for e in range(off + 8, off + block, 2):
            v = struct.unpack_from('<H', raw, e)[0]
            if v >> 12 and in_code(page + (v & 4095)):
                if v >> 12 != 10:
                    raise Refused(f'relocation type {v >> 12} in code')
                stubs.append((page + (v & 4095), e))
        off += block
    blob, sites = b'', 0
    for s in code:
        text = bytearray(raw[s['position']:s['position'] + s['length']])
        for literal, entry in stubs:
            if in_code(literal) is s:
                target = struct.unpack_from('<Q', raw, file_offset(literal))[0] - pe['base']
                if not 0 <= target < pe['size']:
                    raise Refused(f'literal at {literal:#x} points outside the image')
                rewrite_stub(text, s['rva'], literal, target)
                struct.pack_into('<H', raw, entry, 0)        # IMAGE_REL_BASED_ABSOLUTE: skipped
        if trampolines is not None:
            more, n = rewrite_code(text, s['rva'], trampolines + len(blob), slot, tls)
            blob, sites = blob + more, sites + n
        raw[s['position']:s['position'] + s['length']] = text
    report = {'teb_sites': sites, 'stubs': len(stubs), 'trampoline_bytes': len(blob), 'image_size': pe['size'],
              'trampolines': trampolines, 'slot': slot, 'tls': tls}
    return bytes(raw), blob, report


def rewrite_image(image, origin, trampolines, slot, tls='tpidrro_el0'):
    """For a signed wrapper such as build/signed-pe/prepare.py (pull request #497): the x18
    instructions of a PE already mapped by RVA (`image`, a bytearray, its system call
    stubs rewritten by rewrite_pe), with RVA 0 at `origin` and the trampolines and slot
    at addresses in the wrapper. Returns the trampolines and how many instructions, as
    #497's thread_state.rewrite does for ARM64EC."""
    blob, sites = b'', 0
    for s in headers(image)['sections']:
        if s['code']:
            text = bytearray(image[s['rva']:s['rva'] + s['length']])
            more, n = rewrite_code(text, origin + s['rva'], trampolines + len(blob), slot, tls)
            image[s['rva']:s['rva'] + s['length']] = text
            blob, sites = blob + more, sites + n
    return blob, sites


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--trampolines', type=lambda s: int(s, 0))
    parser.add_argument('--slot', type=lambda s: int(s, 0))
    parser.add_argument('--blob', type=Path)
    parser.add_argument('--tls', choices=sorted(MRS), default='tpidrro_el0')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if (args.trampolines is None) != (args.slot is None) or (args.blob and args.trampolines is None):
        parser.error('--trampolines and --slot go together, and --blob needs them')
    try:
        out, blob, report = rewrite_pe(args.input.read_bytes(), args.trampolines, args.slot, args.tls)
    except Refused as error:
        sys.exit(f'{args.input.name}: refused: {error}')
    args.output.write_bytes(out)
    if args.blob:
        args.blob.write_bytes(blob)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(f"{args.input.name}: {report['stubs']} system call stubs made PC-relative"
          + (f", {report['teb_sites']} x18 instructions through trampolines ({len(blob)} bytes)"
             if args.trampolines is not None else ''))


if __name__ == '__main__':
    main()
