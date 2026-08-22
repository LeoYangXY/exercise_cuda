#!/usr/bin/env python3
"""SASS-level ILP for kHand: SSA-rename temps, then space HMMA.

sm_120 instructions are 16 bytes. We only rewrite the inner loop, keeping BRA
at the same address so relative branch offsets stay valid.

ptxas aliases independent epi temps (WAW on R25 etc.). Rename splits those
webs; the DAG is then true RAW plus WAR on pinned acc/A/B/sinks. List-schedule
even HMMA gaps wrapping BRA. REGCOUNT is set to 255 (count is rounded by the
driver; too-tight values make high R indices illegal).

  nvcc -O3 -arch=sm_120 -o /tmp/swp_hand micro_bench/swp_hand.cu -lcuda
  nvcc -O3 -arch=sm_120 -cubin -o /tmp/swp_hand.cubin micro_bench/swp_hand.cu
  python3 micro_bench/patch_sass.py /tmp/swp_hand.cubin kHand -o /tmp/kHand.patched.cubin
  /tmp/swp_hand /tmp/kHand.patched.cubin
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys

INST = 16
NVDISASM = os.environ.get("NVDISASM", "nvdisasm")
RZ = 255

IGNORE = {
    "RZ",
    "URZ",
    "PT",
    "UPT",
    "SRZ",
    "SR_CLOCKLO",
    "SR_CLOCKHI",
    "SR_TID.X",
    "SR_TID.Y",
    "SR_LANEID",
}

# Acc (4×4), A (4), B fragments. IADD3 loop-carried sinks live out of the loop.
PINNED = set()
for _base in (8, 12, 16, 20):  # HMMA dest/C
    PINNED.update(range(_base, _base + 4))
PINNED.update(range(4, 8))  # A
PINNED.update((2, 3, 24, 26, 28))  # B / b1
PINNED.update((48, 49, 51, 52))  # sinks used after BRA


def parse_elf_sections(data: bytes):
    if data[:4] != b"\x7fELF":
        raise SystemExit("not ELF")
    e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
    e_shentsize = struct.unpack_from("<H", data, 0x3A)[0]
    e_shnum = struct.unpack_from("<H", data, 0x3C)[0]
    e_shstrndx = struct.unpack_from("<H", data, 0x3E)[0]
    sh_str = e_shoff + e_shstrndx * e_shentsize
    str_off, str_size = struct.unpack_from("<QQ", data, sh_str + 0x18)
    shstr = data[str_off : str_off + str_size]
    secs = []
    for i in range(e_shnum):
        o = e_shoff + i * e_shentsize
        name_off, typ, flags, addr, off, size = struct.unpack_from("<IIQQQQ", data, o)
        n = shstr[name_off : shstr.find(b"\x00", name_off)].decode()
        secs.append({"name": n, "off": off, "size": size, "addr": addr, "type": typ})
    return secs


def elf_sym_index(data: bytes, name: str) -> int | None:
    e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
    e_shentsize = struct.unpack_from("<H", data, 0x3A)[0]
    e_shnum = struct.unpack_from("<H", data, 0x3C)[0]
    e_shstrndx = struct.unpack_from("<H", data, 0x3E)[0]
    sh_str = e_shoff + e_shstrndx * e_shentsize
    str_off, str_size = struct.unpack_from("<QQ", data, sh_str + 0x18)
    shstr = data[str_off : str_off + str_size]
    sym_off = strtab_off = None
    entsize = 24
    for i in range(e_shnum):
        o = e_shoff + i * e_shentsize
        nm, typ, flags, addr, off, size, link, info, al, es = struct.unpack_from(
            "<IIQQQQIIQQ", data, o
        )
        n = shstr[nm : shstr.find(b"\x00", nm)].decode()
        if n == ".symtab":
            sym_off, sym_size, entsize = off, size, es
        elif n == ".strtab":
            strtab_off = off
    if sym_off is None or strtab_off is None:
        return None
    strtab = data[strtab_off:]
    for i in range(sym_size // entsize):
        st = data[sym_off + i * entsize :]
        st_name = struct.unpack_from("<I", st)[0]
        nm = strtab[st_name : strtab.find(b"\x00", st_name)].decode()
        if nm == name:
            return i
    return None


def nvdisasm_json(cubin: str):
    out = subprocess.check_output([NVDISASM, "--emit-json", cubin], stderr=subprocess.DEVNULL)
    return json.loads(out)


def strip_mod(tok: str) -> str:
    tok = tok.strip()
    tok = tok.split(".", 1)[0]
    return tok


def parse_regs(operands: str | None):
    if not operands:
        return []
    found = re.findall(r"\b(UR\d+|UP\d+|R\d+|P\d+)\b", operands)
    return [f for f in found if f not in IGNORE]


def parse_gprs(operands: str | None) -> list[int]:
    if not operands:
        return []
    return [int(x) for x in re.findall(r"\bR(\d+)\b", operands)]


def is_hmma(op: str) -> bool:
    return op.startswith("HMMA")


def is_uniform(op: str) -> bool:
    return op.startswith("U") or op.startswith("CS2U")


def expand_writes(op: str, operands: str | None, regs: list[str]) -> set[str]:
    w = set()
    if not regs:
        return w
    if is_hmma(op):
        d = regs[0]
        if d.startswith("R"):
            base = int(d[1:])
            for k in range(4):
                w.add(f"R{base + k}")
        else:
            w.add(d)
        return w
    if "SETP" in op or op.startswith("ISETP") or op.startswith("UISETP"):
        if operands:
            first = operands.split(",")[0].strip()
            first = strip_mod(first)
            if first.startswith(("P", "UP")):
                w.add(first)
    dest = regs[0]
    w.add(dest)
    if "64" in op and dest.startswith("R"):
        w.add(f"R{int(dest[1:]) + 1}")
    return w


def expand_reads(op: str, operands: str | None, pred: str | None, regs: list[str]) -> set[str]:
    rset = set()
    if pred:
        p = pred.replace("@", "").replace("!", "")
        if p and p not in IGNORE:
            rset.add(p)
    if not regs:
        return rset
    if is_hmma(op):
        if len(regs) >= 4:
            a, b, c = regs[1], regs[2], regs[3]
            if a.startswith("R"):
                ba = int(a[1:])
                rset.update(f"R{ba + k}" for k in range(4))
            else:
                rset.add(a)
            if b.startswith("R"):
                bb = int(b[1:])
                rset.update(f"R{bb + k}" for k in range(2))
            else:
                rset.add(b)
            if c.startswith("R"):
                bc = int(c[1:])
                rset.update(f"R{bc + k}" for k in range(4))
            else:
                rset.add(c)
        else:
            rset.update(regs)
        return rset
    gprs = parse_gprs(operands)
    dest = gprs[0] if gprs else None
    for r in gprs[1:] if dest is not None else gprs:
        rset.add(f"R{r}")
    for r in regs:
        if r.startswith(("UR", "UP", "P")):
            rset.add(r)
    return rset


def mem_side(op: str) -> str | None:
    if op.startswith(("LDG", "STG", "LDS", "STS", "LDL", "STL", "ATOM", "RED")):
        return "gmem"
    if op.startswith(("BAR", "MEMBAR", "CCTL", "LD.global", "ST.global")):
        return "gmem"
    if op.startswith("LDC"):
        return "cmem"
    return None


def analyze(insts: list[dict]):
    info = []
    for ins in insts:
        op = ins["opcode"]
        ops = ins.get("operands") or ""
        pred = ins.get("predicate")
        regs = parse_regs(ops)
        writes = expand_writes(op, ops, regs)
        reads = expand_reads(op, ops, pred, regs)
        cf = False
        oa = ins.get("other-attributes") or {}
        if isinstance(oa, dict) and oa.get("control-flow"):
            cf = True
        if op.startswith(("BRA", "EXIT", "RET", "CALL")):
            cf = True
        info.append(
            {
                "op": op,
                "ops": ops,
                "pred": pred,
                "reads": reads,
                "writes": writes,
                "mem": mem_side(op),
                "cf": cf,
                "hmma": is_hmma(op),
            }
        )
    return info


def conflicts(a: dict, b: dict) -> bool:
    if a["writes"] & b["reads"] or a["reads"] & b["writes"] or a["writes"] & b["writes"]:
        return True
    if a["mem"] and b["mem"] and a["mem"] == b["mem"]:
        return True
    return False


def conflicts_raw(a: dict, b: dict, pinned: set[str]) -> bool:
    """True dependence + WAR/WAW only on registers we did not split."""
    if a["writes"] & b["reads"]:
        return True
    if a["writes"] & b["writes"] & pinned:
        return True
    if a["reads"] & b["writes"] & pinned:
        return True
    if a["mem"] and b["mem"] and a["mem"] == b["mem"]:
        return True
    return False


def find_loop(insts: list[dict]):
    best = None
    best_score = -1
    for i, ins in enumerate(insts):
        if not ins["opcode"].startswith("BRA"):
            continue
        ops = ins.get("operands") or ""
        m = re.search(r"0x([0-9a-fA-F]+)", ops)
        if not m:
            continue
        tgt = int(m.group(1), 16)
        tidx = tgt // INST
        if tidx < 0 or tidx >= i:
            continue
        nh = sum(1 for j in range(tidx, i) if insts[j]["opcode"].startswith("HMMA"))
        score = nh * 1000 + (i - tidx)
        if score > best_score:
            best_score = score
            best = (tidx, i, tgt)
    return best


def op_latency(op: str) -> int:
    if op.startswith("HMMA"):
        return 34
    if op.startswith("F2I"):
        return 22
    if op.startswith("I2F") or op.startswith("I2FP"):
        return 15
    if op.startswith(("FFMA", "FMUL", "FMNMX", "FADD")):
        return 5
    if op.startswith(("VIMNMX", "IADD", "IMAD", "LOP")):
        return 4
    if op.startswith("MOV"):
        return 1
    return 4


def list_schedule_even(info: list[dict], lo: int, hi: int, gap: int | None, conflict_fn=None,
                       extra_succ: dict | None = None, unlock_hmma: bool = False):
    if conflict_fn is None:
        conflict_fn = conflicts
    body = list(range(lo, hi))
    n = len(body)
    succ = {i: list((extra_succ or {}).get(i, [])) for i in body}
    pred_count = {i: 0 for i in body}
    for i in body:
        for j in succ[i]:
            pred_count[j] += 1
    for a in range(n):
        ia = body[a]
        for b in range(a + 1, n):
            ib = body[b]
            if conflict_fn(info[ia], info[ib]):
                if ib not in succ[ia]:
                    succ[ia].append(ib)
                    pred_count[ib] += 1
    hmma_ids = [i for i in body if info[i]["hmma"]]
    n_hmma = max(len(hmma_ids), 1)
    n_other = n - len(hmma_ids)
    if gap is None:
        gap = max(8, n_other // n_hmma)
    gap = min(gap, max(1, n_other // n_hmma))
    ready = [i for i in body if pred_count[i] == 0]
    ready.sort()
    scheduled = []
    since = 10**9

    reaches_hmma = set()
    if unlock_hmma:
        preds: dict[int, list[int]] = {i: [] for i in body}
        for a, js in succ.items():
            for j in js:
                preds[j].append(a)
        stack = list(hmma_ids)
        seen: set[int] = set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            reaches_hmma.add(x)
            stack.extend(preds[x])

    def unlock(i):
        for j in succ[i]:
            pred_count[j] -= 1
            if pred_count[j] == 0:
                ready.append(j)
        ready.sort()

    while len(scheduled) < n:
        if not ready:
            raise RuntimeError("schedule stuck")
        hmma_ready = [i for i in ready if info[i]["hmma"]]
        other_ready = [i for i in ready if not info[i]["hmma"]]
        if hmma_ready and since >= gap:
            pick = hmma_ready[0]
        elif other_ready:
            if unlock_hmma:
                other_ready.sort(key=lambda i: (0 if i in reaches_hmma else 1, i))
            pick = other_ready[0]
        elif hmma_ready:
            pick = hmma_ready[0]
        else:
            pick = ready[0]
        ready.remove(pick)
        scheduled.append(pick)
        since = 0 if info[pick]["hmma"] else since + 1
        unlock(pick)
    return scheduled, gap, hmma_ids


def list_schedule_lat(info: list[dict], lo: int, hi: int, gap: int | None):
    body = list(range(lo, hi))
    n = len(body)
    succ = {i: [] for i in body}
    pred_count = {i: 0 for i in body}
    for a in range(n):
        ia = body[a]
        for b in range(a + 1, n):
            ib = body[b]
            if conflicts(info[ia], info[ib]):
                succ[ia].append(ib)
                pred_count[ib] += 1

    hmma_ids = [i for i in body if info[i]["hmma"]]
    n_hmma = max(len(hmma_ids), 1)
    n_other = n - len(hmma_ids)
    if gap is None:
        gap = max(8, n_other // n_hmma)
    gap = min(gap, max(1, n_other // n_hmma))

    ready = [i for i in body if pred_count[i] == 0]
    scheduled = []
    ready_at: dict[str, int] = {}
    cycle = 0
    last_hmma_cyc = -32
    modeled_stall = 0

    def unlock(i):
        for j in succ[i]:
            pred_count[j] -= 1
            if pred_count[j] == 0:
                ready.append(j)

    def src_stall(i, c):
        st = 0
        for r in info[i]["reads"]:
            if r in ready_at:
                st = max(st, ready_at[r] - c)
        return max(0, st)

    def prio(i, st, c):
        op = info[i]["op"]
        dist = c - last_hmma_cyc
        tc_ready = dist >= 32
        if info[i]["hmma"] and st == 0 and tc_ready:
            return (0, 0, i)
        if op.startswith("F2I") and st == 0:
            return (1, 0, i)
        if op.startswith(("FFMA", "FMUL", "FMNMX", "FADD")) and st == 0:
            return (2, 0, i)
        if info[i]["hmma"] and st == 0:
            return (3, max(0, 32 - dist), i)
        if info[i]["hmma"]:
            return (6, st + max(0, 32 - dist), i)
        if op.startswith(("VIMNMX", "IADD")):
            return (5, st, i)
        return (4, st, i)

    while len(scheduled) < n:
        if not ready:
            raise RuntimeError("schedule stuck (cycle in dep graph)")
        scored = []
        for i in ready:
            st = src_stall(i, cycle)
            scored.append((prio(i, st, cycle), st, i))
        scored.sort()
        _, st, pick = scored[0]
        ready.remove(pick)
        if st:
            modeled_stall += st
            cycle += st
        scheduled.append(pick)
        lat = op_latency(info[pick]["op"])
        for r in info[pick]["writes"]:
            ready_at[r] = cycle + lat
        if info[pick]["hmma"]:
            last_hmma_cyc = cycle
        cycle += 1
        unlock(pick)

    print(f"  modeled issue cycles={cycle}  scoreboard bubbles={modeled_stall}  "
          f"cyc/mma={cycle / n_hmma:.2f}")
    return scheduled, gap, hmma_ids


def apply_perm(blob: bytes, lo: int, hi: int, order: list[int]) -> bytes:
    insts = [blob[i * INST : (i + 1) * INST] for i in range(len(blob) // INST)]
    new = insts[:]
    for off, src in enumerate(order):
        new[lo + off] = insts[src]
    return b"".join(new) + blob[len(new) * INST :]


def get_stall(b: bytes) -> int:
    inst = int.from_bytes(b, "little")
    return (inst >> 105) & 0xF


def set_stall(b: bytes, stall: int) -> bytes:
    inst = int.from_bytes(b, "little")
    inst = (inst & ~(0xF << 105)) | ((stall & 0xF) << 105)
    return inst.to_bytes(16, "little")


def get_wrbar(b: bytes) -> int:
    return (int.from_bytes(b, "little") >> 110) & 7


def set_wrbar(b: bytes, wr: int) -> bytes:
    inst = int.from_bytes(b, "little")
    inst = (inst & ~(0x7 << 110)) | ((wr & 7) << 110)
    return inst.to_bytes(16, "little")


def get_wait(b: bytes) -> int:
    return (int.from_bytes(b, "little") >> 116) & 0x3F


def set_wait(b: bytes, wait: int) -> bytes:
    inst = int.from_bytes(b, "little")
    inst = (inst & ~(0x3F << 116)) | ((wait & 0x3F) << 116)
    return inst.to_bytes(16, "little")


def clear_reuse(b: bytes) -> bytes:
    inst = int.from_bytes(b, "little")
    inst &= ~(0xF << 122)
    return inst.to_bytes(16, "little")


def set_field(inst: int, bit: int, val: int, width: int = 8) -> int:
    mask = (1 << width) - 1
    return (inst & ~(mask << bit)) | ((val & mask) << bit)


def rewrite_stalls(blob: bytes, info: list[dict], lo: int, hi: int) -> bytes:
    ninst = len(blob) // INST
    out = [blob[i * INST : (i + 1) * INST] for i in range(ninst)]
    changed = 0
    for i in range(lo, hi):
        nxt = i + 1
        if nxt >= len(info):
            break
        if info[i]["writes"] & info[nxt]["reads"]:
            continue
        old = get_stall(out[i])
        if old > 1:
            out[i] = set_stall(out[i], 1)
            changed += 1
    print(f"  rewrote {changed} stall fields -> 1 (independent successor)")
    return b"".join(out)


# Operand bit slots (empirically from this sm_120 cubin / Hopper-style 8-bit fields).
# dest is always bits 16-23 for these ALU/HMMA forms.
def gpr_slots(op: str, gprs: list[int]) -> list[tuple[int, int, str]]:
    """(bit, old_reg, 'd'|'s') for each GPR field we are allowed to rewrite."""
    if not gprs:
        return []
    if op.startswith("U") or op.startswith("CS2U") or op.startswith("BRA"):
        return []
    dest = gprs[0]
    srcs = gprs[1:]
    if is_hmma(op):
        # dest 16, A 24, B 32, C 64. bits 72-79 are shape (0x18), not a reg.
        out = [(16, dest, "d"), (24, srcs[0], "s"), (32, srcs[1], "s"), (64, srcs[2], "s")]
        return out
    if op.startswith("FFMA"):
        # Rd, Ra, Rb, fimm  — Rb lives in w1[0:8], imm in bits 32-63
        if len(srcs) < 2:
            return [(16, dest, "d")] + [(24, s, "s") for s in srcs]
        return [(16, dest, "d"), (24, srcs[0], "s"), (64, srcs[1], "s")]
    if op.startswith("FMUL") or op.startswith("FADD"):
        return [(16, dest, "d")] + ([(24, srcs[0], "s")] if srcs else [])
    if op.startswith("FMNMX"):
        # Rd, RZ, Ra  — RZ is 255 at bits 24-31, Ra at 32
        if len(srcs) >= 1:
            return [(16, dest, "d"), (32, srcs[0], "s")]
        return [(16, dest, "d")]
    if op.startswith("F2I") or op.startswith("I2F"):
        return [(16, dest, "d")] + ([(32, srcs[0], "s")] if srcs else [])
    if op.startswith("VIMNMX"):
        return [(16, dest, "d")] + ([(24, srcs[0], "s")] if srcs else [])
    if op.startswith("IADD3"):
        # Rd, (PT,PT,) Ra, Rb, Rc  — Rc at bit 64
        slots = [(16, dest, "d")]
        bits = (24, 32, 64)
        for b, s in zip(bits, srcs):
            slots.append((b, s, "s"))
        return slots
    if op.startswith("IADD"):
        return [(16, dest, "d")] + ([(24, srcs[0], "s")] if srcs else [])
    if op.startswith("MOV"):
        # MOV Rd, imm  → only dest. MOV Rd, Ra → dest 16, src 32
        if srcs:
            return [(16, dest, "d"), (32, srcs[0], "s")]
        return [(16, dest, "d")]
    # fallback: dest only
    return [(16, dest, "d")]


def patch_inst_regs(raw: bytes, op: str, ops: str, dest_new: int | None,
                    src_new: list[int] | None) -> bytes:
    gprs = parse_gprs(ops)
    if not gprs or is_uniform(op) or op.startswith("BRA"):
        return raw
    slots = gpr_slots(op, gprs)
    inst = int.from_bytes(raw, "little")
    si = 0
    for bit, old, kind in slots:
        if kind == "d":
            if dest_new is None:
                continue
            cur = (inst >> bit) & 0xFF
            if cur != old and cur != RZ:
                # already patched or unexpected encoding — still force dest
                pass
            inst = set_field(inst, bit, dest_new)
        else:
            if src_new is None or si >= len(src_new):
                si += 1
                continue
            inst = set_field(inst, bit, src_new[si])
            si += 1
    return inst.to_bytes(16, "little")


def ssa_rename(info: list[dict], lo: int, hi: int, pool_start: int = 80, pool_end: int = 255,
               only_ops: set[str] | None = None):
    """Split WAW on non-pinned GPRs. Loop-carried last-def writes the live-in name."""
    body = list(range(lo, hi))
    writes_of: dict[int, list[int]] = {}
    first_use_before_def: set[int] = set()
    seen_def: set[int] = set()
    for i in body:
        gprs = parse_gprs(info[i]["ops"])
        if is_uniform(info[i]["op"]) or info[i]["op"].startswith("BRA"):
            continue
        srcs = gprs[1:] if gprs else []
        dest = gprs[0] if gprs else None
        for s in srcs:
            if s not in seen_def:
                first_use_before_def.add(s)
        if dest is not None:
            writes_of.setdefault(dest, []).append(i)
            seen_def.add(dest)

    carried = {r for r in first_use_before_def if r not in PINNED}

    used = set(PINNED)
    for i in body:
        used.update(parse_gprs(info[i]["ops"]))
    next_reg = pool_start

    def alloc() -> int:
        nonlocal next_reg
        while next_reg in used or next_reg in PINNED or next_reg == RZ:
            next_reg += 1
        if next_reg >= pool_end:
            for r in range(pool_start, pool_end):
                if r not in PINNED and r != RZ:
                    next_reg = r + 1
                    used.add(r)
                    return r
            raise RuntimeError("out of registers")
        r = next_reg
        used.add(r)
        next_reg += 1
        return r

    last_phys = {r: r for r in range(256)}
    write_idx = {r: 0 for r in writes_of}
    maps = {}  # inst -> (dest_new, src_new)

    for i in body:
        op = info[i]["op"]
        ops = info[i]["ops"]
        gprs = parse_gprs(ops)
        if not gprs or is_uniform(op) or op.startswith("BRA"):
            maps[i] = (None, None)
            continue
        dest = gprs[0]
        srcs = gprs[1:]
        src_new = [last_phys[s] for s in srcs]
        nwrites = len(writes_of.get(dest, []))
        wseq = write_idx.get(dest, 0)
        is_last = wseq == nwrites - 1
        if dest in PINNED or is_hmma(op):
            dest_new = dest
        elif only_ops is not None and not any(op.startswith(p) for p in only_ops):
            dest_new = dest
        elif dest in carried and is_last:
            dest_new = dest
        elif wseq == 0 and dest not in carried:
            dest_new = dest
        else:
            dest_new = alloc()
        last_phys[dest] = dest_new
        write_idx[dest] = wseq + 1
        maps[i] = (dest_new, src_new)

    # Wrap: last_phys of a carried reg must equal the live-in name.
    bad = [r for r in carried if last_phys[r] != r]
    if bad:
        print(f"  warning: carried regs not closed {bad[:12]}...")

    n_split = sum(1 for i in body if maps[i][0] is not None
                  and parse_gprs(info[i]["ops"])
                  and maps[i][0] != parse_gprs(info[i]["ops"])[0])
    print(f"  SSA: split {n_split} dests  maxR={max(used)}  carried={sorted(carried)[:16]}...")
    return maps, max(used), carried


def rebuild_info_mapped(info: list[dict], maps: dict, lo: int, hi: int) -> list[dict]:
    out = []
    for i, inf in enumerate(info):
        if i < lo or i >= hi:
            out.append(inf)
            continue
        dest_new, src_new = maps[i]
        op = inf["op"]
        pred = inf["pred"]
        gprs = parse_gprs(inf["ops"])
        reads: set[str] = set()
        writes: set[str] = set()
        if pred:
            p = pred.replace("@", "").replace("!", "")
            if p and p not in IGNORE:
                reads.add(p)
        if is_uniform(op) or not gprs:
            out.append(inf)
            continue
        dest = dest_new if dest_new is not None else gprs[0]
        srcs = src_new if src_new is not None else gprs[1:]
        if is_hmma(op):
            base = dest
            writes.update(f"R{base + k}" for k in range(4))
            if len(srcs) >= 3:
                a, b, c = srcs[0], srcs[1], srcs[2]
                reads.update(f"R{a + k}" for k in range(4))
                reads.update(f"R{b + k}" for k in range(2))
                reads.update(f"R{c + k}" for k in range(4))
        else:
            writes.add(f"R{dest}")
            for s in srcs:
                reads.add(f"R{s}")
            for r in parse_regs(inf["ops"]):
                if r.startswith(("UR", "UP", "P")):
                    if r in inf["writes"]:
                        writes.add(r)
                    else:
                        reads.add(r)
        new = dict(inf)
        new["reads"] = reads
        new["writes"] = writes
        out.append(new)
    return out


def retarget_f2i_barriers(blob: bytes, info: list[dict], lo: int, hi: int) -> bytes:
    """Give in-flight F2Is distinct wrbars; consumers wait if cyclic dist < L."""
    ninst = len(blob) // INST
    out = [blob[i * INST : (i + 1) * INST] for i in range(ninst)]
    n = hi - lo
    f2i = [i for i in range(lo, hi) if info[i]["op"].startswith("F2I")]
    if not f2i:
        return blob
    L = 22
    # producer dest -> index
    prod = {}
    for i in f2i:
        for r in info[i]["writes"]:
            prod[r] = i
    # consumer list per F2I
    cons: dict[int, list[int]] = {i: [] for i in f2i}
    for j in range(lo, hi):
        for r in info[j]["reads"]:
            if r in prod and prod[r] != j:
                cons[prod[r]].append(j)

    def cyclic(a, b):
        pa, pb = a - lo, b - lo
        d = pb - pa
        return d if d > 0 else d + n

    # live interval [pos, pos+hold)
    intervals = []
    for i in f2i:
        hold = L
        for c in cons[i]:
            hold = max(hold, cyclic(i, c))
        intervals.append((i, hold))

    color = {}
    active = []  # (end_cyc, bar, idx)
    bars = list(range(6))
    for i, hold in intervals:
        pos = i - lo
        active = [(e, b, k) for e, b, k in active if e > pos]
        used = {b for _, b, _ in active}
        pick = next((b for b in bars if b not in used), None)
        if pick is None:
            pick = pos % 6
        color[i] = pick
        active.append((pos + hold, pick, i))

    # clear waits in body, then set wrbar + consumer waits
    for i in range(lo, hi):
        out[i] = set_wait(out[i], 0)
        if get_wrbar(out[i]) != 7 and not info[i]["op"].startswith("F2I"):
            # leave non-F2I wrbar alone (usually 7)
            pass
    for i in f2i:
        out[i] = set_wrbar(out[i], color[i])
        out[i] = set_wait(out[i], 0)
    waits = {i: 0 for i in range(lo, hi)}
    for i in f2i:
        bit = 1 << color[i]
        for c in cons[i]:
            if cyclic(i, c) < L + 2:
                waits[c] |= bit
        # loop-head consumers already in cons via wrap RAW... but wrap RAW is
        # not in the intra-iter graph. Add waits for dest uses before this F2I.
        dests = info[i]["writes"]
        for j in range(lo, i):
            if info[j]["reads"] & dests:
                if cyclic(i, j) < L + 2:
                    waits[j] |= bit
    for j, m in waits.items():
        if m:
            out[j] = set_wait(out[j], m)

    print(f"  F2I barriers: {len(f2i)} insts colored {sorted(set(color.values()))}  "
          f"waits set on {sum(1 for m in waits.values() if m)} insts")
    return b"".join(out)


def dump_gaps(info, order, lo, label):
    seq = order
    h = [k for k, i in enumerate(seq) if info[i]["hmma"]]
    print(f"  [{label}] HMMA at body+{h}  dests:",
          [info[seq[k]]["ops"].split(",")[0] for k in h])
    if len(h) >= 2:
        gaps = [h[i + 1] - h[i] - 1 for i in range(len(h) - 1)]
        tail = len(seq) - h[-1] - 1
        print(f"  [{label}] gaps between HMMA: {gaps}  after last: {tail}  "
              f"wrap={tail + h[0]}")
    ops = []
    for i in seq:
        o = info[i]["op"]
        if info[i]["hmma"]:
            ops.append("HMMA")
        elif o.startswith("F2I"):
            ops.append("F2I")
        elif o.startswith("FFMA"):
            ops.append("FFMA")
        elif o.startswith("FMNMX") or o.startswith("FMUL"):
            ops.append(o.split(".")[0][:5])
        elif o.startswith("VIMNMX"):
            ops.append("IMNMX")
        elif o.startswith("IADD"):
            ops.append("IADD")
        elif o.startswith("BRA"):
            ops.append("BRA")
        else:
            ops.append(".")
    print("  stream:", " ".join(ops))


def update_regcount(data: bytearray, kernel: str, new_n: int) -> None:
    idx = elf_sym_index(bytes(data), kernel)
    if idx is None:
        print("  warn: no symbol for regcount")
        return
    needle = bytes([0x04, 0x2F, 0x08, 0x00]) + struct.pack("<I", idx)
    nrep = 0
    start = 0
    while True:
        p = data.find(needle, start)
        if p < 0:
            break
        old = struct.unpack_from("<I", data, p + 8)[0]
        if new_n > old:
            struct.pack_into("<I", data, p + 8, new_n)
            nrep += 1
            print(f"  REGCOUNT sym {idx} {old} -> {new_n} at {p:#x}")
        start = p + 1
    if nrep == 0:
        print(f"  REGCOUNT unchanged (needles for sym {idx}: searched)")


def patch_cubin(src: str, dst: str, kernel: str, gap: int | None, dry: bool, mode: str = "rename",
                no_perm: bool = False, no_barrier: bool = False, no_stall: bool = False,
                no_reuse_clear: bool = False, pool_start: int = 80, pool_end: int = 255,
                only_ops: set[str] | None = None):
    data = bytearray(open(src, "rb").read())
    secs = parse_elf_sections(data)
    j = nvdisasm_json(src)
    fns = j[1]
    fn = None
    for f in fns:
        if f["function-name"] == kernel or f["function-name"].startswith(kernel):
            fn = f
            break
    if not fn:
        names = [f["function-name"] for f in fns]
        raise SystemExit(f"kernel {kernel} not found. have {names}")
    insts = fn["sass-instructions"]
    info = analyze(insts)
    loop = find_loop(insts)
    if not loop:
        raise SystemExit("no inner loop BRA found")
    lo, hi, tgt = loop
    print(f"kernel {fn['function-name']}  ninst={len(insts)}  loop [{lo},{hi})  BRA->{tgt:#x}")
    hmma = [i for i in range(lo, hi) if info[i]["hmma"]]
    print(f"  loop HMMA indices {hmma} count={len(hmma)}")
    if len(hmma) < 2:
        raise SystemExit("need >= 2 HMMA in the loop to interleave")

    sec = None
    for s in secs:
        if s["name"] == f".text.{fn['function-name']}" or s["name"].endswith(fn["function-name"]):
            if s["name"].startswith(".text"):
                sec = s
                break
    if not sec:
        for s in secs:
            print(" section", s["name"])
        raise SystemExit("text section not found")
    blob = bytes(data[sec["off"] : sec["off"] + sec["size"]])
    if len(blob) < hi * INST:
        raise SystemExit(f"text too small {len(blob)} vs {hi * INST}")

    maxr = 0
    sched_info = info
    force_last = [i for i in range(lo, hi)
                  if info[i]["op"].startswith("UIADD3") and "URZ" in (info[i]["ops"] or "")]

    if mode == "rename":
        maps, maxr, carried = ssa_rename(info, lo, hi, pool_start=pool_start, pool_end=pool_end,
                                only_ops=only_ops)
        inst_list = [blob[i * INST : (i + 1) * INST] for i in range(len(blob) // INST)]
        for i in range(lo, hi):
            if not no_reuse_clear:
                inst_list[i] = clear_reuse(inst_list[i])
            dest_new, src_new = maps[i]
            inst_list[i] = patch_inst_regs(
                inst_list[i], info[i]["op"], info[i]["ops"], dest_new, src_new
            )
        blob = b"".join(inst_list)
        sched_info = rebuild_info_mapped(info, maps, lo, hi)
        pinned_names = {f"R{r}" for r in PINNED}
        # glue reconvergence NOP to the end of the body
        extra_succ = {i: [] for i in range(lo, hi)}
        if force_last:
            rec = force_last[-1]
            for i in range(lo, hi):
                if i != rec:
                    extra_succ[i].append(rec)

        def cfn(a, b, _p=pinned_names):
            return conflicts_raw(a, b, _p)

        order, used_gap, _ = list_schedule_even(
            sched_info, lo, hi, gap, conflict_fn=cfn, extra_succ=extra_succ
        )
        print(f"  mode=rename target gap={used_gap}")
    elif mode == "even":
        order, used_gap, _ = list_schedule_even(info, lo, hi, gap)
        print(f"  mode={mode} target gap={used_gap}")
    else:
        order, used_gap, _ = list_schedule_lat(info, lo, hi, gap)
        print(f"  mode={mode} target gap={used_gap}")

    dump_gaps(info, list(range(lo, hi)), lo, "before")
    dump_gaps(sched_info, order, lo, "after")

    if sorted(order) != list(range(lo, hi)):
        raise SystemExit("scheduler is not a permutation")

    if no_perm:
        order = list(range(lo, hi))
        print("  skip permute (rename only)")
    new_blob = apply_perm(blob, lo, hi, order)
    if len(new_blob) != len(blob):
        raise SystemExit("size changed")
    new_info = sched_info[:]
    for off, src in enumerate(order):
        new_info[lo + off] = sched_info[src]
    if mode == "rename" and not no_barrier:
        new_blob = retarget_f2i_barriers(new_blob, new_info, lo, hi)
    if not no_stall:
        new_blob = rewrite_stalls(new_blob, new_info, lo, hi)
    if dry:
        print("dry-run, not writing")
        return
    data[sec["off"] : sec["off"] + sec["size"]] = new_blob
    if mode == "rename" and maxr:
        # Hardware/driver rounds the cubin REGCOUNT; too-tight values make
        # R_n an illegal instruction (R78 failed at count 80, worked at 96).
        rounded = 255
        update_regcount(data, fn["function-name"], rounded)
    open(dst, "wb").write(data)
    print(f"wrote {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cubin")
    ap.add_argument("kernel", nargs="?", default="kHand")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--mode", choices=("rename", "even", "lat"), default="rename")
    ap.add_argument("--gap", type=int, default=27)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--no-perm", action="store_true")
    ap.add_argument("--no-barrier", action="store_true", help="keep original F2I barrier fields")
    ap.add_argument("--no-stall", action="store_true")
    ap.add_argument("--no-reuse-clear", action="store_true")
    ap.add_argument("--pool-start", type=int, default=80)
    ap.add_argument("--pool-end", type=int, default=255)
    ap.add_argument("--only", default=None, help="comma-separated opcode prefixes to rename")
    args = ap.parse_args()
    dst = args.output or args.cubin.replace(".cubin", ".patched.cubin")
    patch_cubin(args.cubin, dst, args.kernel, args.gap, args.dry, args.mode,
                no_perm=args.no_perm, no_barrier=args.no_barrier, no_stall=args.no_stall,
                no_reuse_clear=args.no_reuse_clear,
                pool_start=args.pool_start, pool_end=args.pool_end,
                only_ops=set(args.only.split(",")) if args.only else None)


if __name__ == "__main__":
    main()
