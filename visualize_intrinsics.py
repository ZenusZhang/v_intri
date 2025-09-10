#!/usr/bin/env python3
import argparse
import math
import os
import re
import sys
from typing import Dict, List, Set, Tuple, Optional
import json
import subprocess
import shutil


SAMPLE_TEXT = r"""
#define COS_FLOAT32(lmul, mlen)                                                \
    inline vfloat32m##lmul##_t cos_float32(const vfloat32m##lmul##_t &v,       \
                                           const size_t vl) {                  \
        auto n = __riscv_vfmv_v_f_f32m##lmul(0x1.45f306p-2f, vl);              \
        auto half = __riscv_vfmv_v_f_f32m##lmul(0.5f, vl);                     \
                                                                               \
        /*  n = rint((|x|+pi/2)/pi) - 0.5. */                                  \
        auto r = __riscv_vfabs_v_f32m##lmul(v, vl);                            \
        n = __riscv_vfmadd_vv_f32m##lmul(n, r, half, vl);                      \
        auto ni = __riscv_vfcvt_x_f_v_i32m##lmul(n, vl);                       \
        n = __riscv_vfcvt_f_x_v_f32m##lmul(ni, vl);                            \
        auto odd = __riscv_vadd_vx_i32m##lmul(ni, 0x1.8p+23, vl);              \
        n = __riscv_vfsub_vf_f32m##lmul(n, 0.5f, vl);                          \
        odd = __riscv_vsll_vx_i32##m##lmul(odd, 31, vl);                       \
                                                                               \
        /* r = |x| - n*pi  (range reduction into -pi/2 .. pi/2).  */           \
        r = __riscv_vfnmsac_vf_f32m##lmul(r, 0x1.921fb6p+1f, n, vl);           \
        r = __riscv_vfnmsac_vf_f32m##lmul(r, -0x1.777a5cp-24f, n, vl);         \
        r = __riscv_vfnmsac_vf_f32m##lmul(r, -0x1.ee59dap-49f, n, vl);         \
                                                                               \
    /* y = sin(r). Implemented using Horner's method. */                       \
    auto r2 = __riscv_vfmul_vv_f32m##lmul(r, r, vl);                           \
    auto r3 = __riscv_vfmul_vv_f32m##lmul(r2, r, vl);                          \
                                                                              \
    /* Define polynomial coefficients for sin(r)/r ≈ 1 + P(r^2) */            \
    const float c9 = 0x1.5b2e76p-19f; /* 1/9! */                              \
    const float c7 = -0x1.9f42eap-13f;/* -1/7! */                              \
    const float c5 = 0x1.110df4p-7f;  /* 1/5! */                              \
    const float c3 = -0x1.555548p-3f; /* -1/3! */                              \
                                                                              \
    /* Evaluate P(r^2) from the innermost term outwards using FMA */          \
    /* y = C9 */                                                              \
    auto y = __riscv_vfmv_v_f_f32m##lmul(c9, vl);                              \
    /* y = y*r2 + C7 */                                                       \
    auto vc7 = __riscv_vfmv_v_f_f32m##lmul(c7, vl); \
    y = __riscv_vfmadd_vv_f32m##lmul(y, r2, vc7, vl); \
    /* y = y*r2 + C5 */                                                       \
    auto vc5 = __riscv_vfmv_v_f_f32m##lmul(c5, vl); \
    y = __riscv_vfmadd_vv_f32m##lmul(y, r2, vc5, vl); \
    /* y = y*r2 + C3 -> Now y holds the full polynomial P(r^2) */             \
    auto vc3 = __riscv_vfmv_v_f_f32m##lmul(c3, vl); \
    y = __riscv_vfmadd_vv_f32m##lmul(y, r2, vc3, vl); \
                                                                              \
    /* Final result: y = r + r3 * P(r^2) */                                   \
    y = __riscv_vfmadd_vv_f32m##lmul(y, r3, r, vl);                            \
                                                                              \
    /* Apply sign correction based on the quadrant from range reduction */    \
    auto tmp = __riscv_vreinterpret_v_f32m##lmul##_i32m##lmul(y);              \
    tmp = __riscv_vxor_vv_i32m##lmul(tmp, odd, vl);                            \
    return __riscv_vreinterpret_v_i32m##lmul##_f32m##lmul(tmp);                \
    }
"""


def normalize_code(text: str) -> str:
    # Remove block and line comments first (preserve newlines for macro handling)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*", "", text)
    # Collapse macro continuations: remove backslash-newline across the whole text
    text = re.sub(r"\\\n", "", text)
    # Drop token-pasting from macros to simplify parsing
    text = text.replace("##", "")
    # If the function is defined inside a macro, strip the `#define NAME(args)` prefix but keep the rest of the line
    text = re.sub(r"^\s*#\s*define\s+\w+\s*\([^)]*\)\s*", "", text, flags=re.MULTILINE)
    # Now remove any remaining preprocessor lines entirely
    text = re.sub(r"^\s*#.*$", "", text, flags=re.MULTILINE)
    return text


def extract_params(text: str) -> Set[str]:
    m = re.search(r"cos_float32\s*\((.*?)\)", text, flags=re.DOTALL)
    if not m:
        return set()
    params = m.group(1)
    parts = [p.strip() for p in params.split(',') if p.strip()]
    names: Set[str] = set()
    for p in parts:
        # Remove & and * and qualifiers
        p2 = p.replace('&', ' ').replace('*', ' ')
        # Take the last identifier as the variable name
        mm = re.search(r"([A-Za-z_]\w*)\s*$", p2)
        if mm:
            candidate = mm.group(1)
            if candidate not in {"const"}:
                names.add(candidate)
    return names


def extract_statements(text: str) -> List[str]:
    # Only consider function body between { and }
    body_m = re.search(r"cos_float32\s*\([^\)]*\)\s*\{(.*)\}", text, flags=re.DOTALL)
    body = body_m.group(1) if body_m else text
    # Split by semicolons
    stmts = [s.strip() for s in body.split(";")]
    # Remove empties
    return [s for s in stmts if s]


def extract_lhs_variables(statements: List[str]) -> Set[str]:
    lhs_vars: Set[str] = set()
    for s in statements:
        if s.startswith("return "):
            continue
        m = re.search(r"(?:^|\s)(?:auto|const)?\s*([A-Za-z_]\w*)\s*=", s)
        if m:
            lhs_vars.add(m.group(1))
    return lhs_vars


def extract_rhs_vars(rhs: str, candidate_vars: Set[str]) -> Set[str]:
    # Collect all identifiers in rhs and filter by known candidate vars
    tokens = set(re.findall(r"\b([A-Za-z_]\w*)\b", rhs))
    # Remove obvious non-vars
    remove = {"auto", "const", "return", "inline", "cos_float32"}
    # Remove intrinsics names starting with __riscv_
    tokens = {t for t in tokens if not t.startswith("__riscv_")}
    tokens = tokens - remove
    # Only keep tokens that are known vars
    return tokens & candidate_vars


def parse_intrinsic_op(rhs: str) -> Optional[str]:
    """Extract an intrinsic op mnemonic from RHS, e.g. '__riscv_vfabs_v_...' -> 'fabs'."""
    m = re.search(r"__riscv_([A-Za-z0-9]+)", rhs)
    if not m:
        return None
    op = m.group(1)
    # Drop leading vector 'v' for readability if present
    if op.startswith('v') and len(op) > 1:
        op = op[1:]
    # Keep only the stem before any underscore suffixes like '_vf_f32m...'
    op = op.split('_', 1)[0]
    return op


def parse_intrinsic_args(rhs: str, candidate_vars: Set[str], excludes: Set[str]) -> List[str]:
    """Parse __riscv_* call arguments robustly, handling nested calls.
    Preserves duplicates and order, filters by candidates/excludes."""

    def split_top_level_commas(s: str) -> List[str]:
        parts: List[str] = []
        depth = 0
        cur: List[str] = []
        for ch in s:
            if ch == '(':
                depth += 1
            elif ch == ')':
                if depth > 0:
                    depth -= 1
            elif ch == ',' and depth == 0:
                parts.append(''.join(cur).strip())
                cur = []
                continue
            cur.append(ch)
        if cur:
            parts.append(''.join(cur).strip())
        return parts

    m = re.search(r"__riscv_[A-Za-z0-9_]+\s*\((.*)\)$", rhs)
    if not m:
        toks = re.findall(r"\b([A-Za-z_]\w*)\b", rhs)
        return [t for t in toks if t in candidate_vars and t not in excludes]
    inner = m.group(1)
    parts = split_top_level_commas(inner)
    result: List[str] = []
    for p in parts:
        # Take the last identifier within this argument
        ids = re.findall(r"([A-Za-z_]\w*)", p)
        if not ids:
            continue
        name = ids[-1]
        if name in candidate_vars and name not in excludes:
            result.append(name)
    return result


def build_edges(statements: List[str], params: Set[str], excludes: Optional[Set[str]] = None) -> Tuple[Set[str], Set[Tuple[str, str]], List[Tuple[str, str, Optional[str]]]]:
    excludes = excludes or set()
    lhs_vars = extract_lhs_variables(statements)
    # Candidate variable names include LHSes, params, and a synthetic 'return'
    candidates: Set[str] = set(lhs_vars) | set(params) | {"return"}

    edges: Set[Tuple[str, str]] = set()
    edges_with_ops: List[Tuple[str, str, Optional[str]]] = []
    nodes: Set[str] = set(candidates)

    for s in statements:
        s = s.strip()
        if not s:
            continue
        if s.startswith("return "):
            rhs = s[len("return "):]
            rhs_vars = parse_intrinsic_args(rhs, candidates, excludes)
            op = parse_intrinsic_op(rhs)
            for rv in rhs_vars:
                edges.add((rv, "return"))
                edges_with_ops.append((rv, "return", op))
                nodes.add(rv)
            nodes.add("return")
            continue

        m = re.search(r"(?:^|\s)(?:auto|const)?\s*([A-Za-z_]\w*)\s*=\s*(.*)$", s)
        if not m:
            continue
        lhs, rhs = m.group(1), m.group(2)
        nodes.add(lhs)
        rhs_vars = parse_intrinsic_args(rhs, candidates, excludes)
        op = parse_intrinsic_op(rhs)
        # In case rhs introduces new vars used before declared (unlikely here), accept them too
        for rv in rhs_vars:
            edges.add((rv, lhs))
            edges_with_ops.append((rv, lhs, op))
            nodes.add(rv)

    return nodes, edges, edges_with_ops


def build_edges_ssa(statements: List[str], params: Set[str], excludes: Optional[Set[str]] = None) -> Tuple[Set[str], Set[Tuple[str, str]], List[Tuple[str, str, Optional[str]]], Dict[str, int]]:
    """SSA-like renaming to break cycles: each assignment to X creates X_fK as a new version.
    Edges go from current version of sources to the new version of sink."""
    excludes = excludes or set()
    lhs_vars = extract_lhs_variables(statements)
    candidates: Set[str] = set(lhs_vars) | set(params) | {"return"}

    current_name: Dict[str, str] = {v: v for v in candidates if v != "return"}
    assign_count: Dict[str, int] = {}

    edges_set: Set[Tuple[str, str]] = set()
    edges_with_ops: List[Tuple[str, str, Optional[str]]] = []
    nodes: Set[str] = set(current_name.values()) | {"return"}
    node_time: Dict[str, int] = {}

    for idx, s in enumerate(statements):
        s = s.strip()
        if not s:
            continue
        if s.startswith("return "):
            rhs = s[len("return "):]
            op = parse_intrinsic_op(rhs)
            rhs_vars = parse_intrinsic_args(rhs, set(current_name.keys()) | {"return"}, excludes)
            for rv in rhs_vars:
                src = current_name.get(rv, rv)
                edges_set.add((src, "return"))
                edges_with_ops.append((src, "return", op))
                nodes.add(src)
            nodes.add("return")
            node_time["return"] = idx
            continue

        m = re.search(r"(?:^|\s)(?:auto|const)?\s*([A-Za-z_]\w*)\s*=\s*(.*)$", s)
        if not m:
            continue
        lhs, rhs = m.group(1), m.group(2)
        op = parse_intrinsic_op(rhs)
        rhs_vars = parse_intrinsic_args(rhs, set(current_name.keys()), excludes)

        # Create new version for lhs
        assign_count[lhs] = assign_count.get(lhs, 0) + 1
        sink = f"{lhs}_f{assign_count[lhs]}"
        nodes.add(sink)
        node_time[sink] = idx

        for rv in rhs_vars:
            src = current_name.get(rv, rv)
            edges_set.add((src, sink))
            edges_with_ops.append((src, sink, op))
            nodes.add(src)

        current_name[lhs] = sink
    return nodes, edges_set, edges_with_ops, node_time


def compute_liveness_ssa(nodes: Set[str],
                         edges: Set[Tuple[str, str]],
                         node_time: Dict[str, int],
                         include_params: bool = False) -> Tuple[Dict[str, Tuple[int, int]], Dict[int, List[str]]]:
    """Compute live intervals per SSA node and the set of alive nodes at each time.
    - Interval for node u is [t_def(u), max_t_use(u)], inclusive.
    - Nodes without uses are considered alive only at their def time.
    - By default, parameters or nodes without a definition time are excluded unless include_params=True.
    Returns (intervals, alive_by_time)."""

    # Collect use times for each node
    uses: Dict[str, List[int]] = {}
    for u, v in edges:
        # Ignore scalar registers (s_*) as sources for liveness; we track vector regs only
        if isinstance(u, str) and u.startswith("s_"):
            continue
        t_def = node_time.get(u, -1)
        t_use = node_time.get(v, -1)
        if v == "return":
            t_use = node_time.get("return", t_use)
        if not include_params and t_def < 0:
            continue
        if t_use < 0:
            # If use time is unknown, skip this use
            continue
        uses.setdefault(u, []).append(t_use)

    # Build intervals
    intervals: Dict[str, Tuple[int, int]] = {}
    for n in nodes:
        # Skip scalar registers in intervals
        if isinstance(n, str) and n.startswith("s_"):
            continue
        t_def = node_time.get(n, -1)
        if not include_params and t_def < 0:
            continue
        if t_def < 0:
            # include_params=True path: treat as alive across all known times
            continue
        last_use = max(uses.get(n, [t_def]))
        intervals[n] = (t_def, last_use)

    # Build alive set per time
    all_times = [t for t in node_time.values() if t >= 0]
    if not all_times:
        return intervals, {}
    min_t, max_t = min(all_times), max(all_times)
    alive_by_time: Dict[int, List[str]] = {}
    for t in range(min_t, max_t + 1):
        alive = [n for n, (s, e) in intervals.items() if s <= t <= e]
        alive_by_time[t] = sorted(alive)
    return intervals, alive_by_time


def rank_nodes(nodes: Set[str], edges: Set[Tuple[str, str]], inputs: Set[str]) -> Dict[str, int]:
    # Longest-path style ranking: initialize inputs to 0, others to 0 as well
    rank: Dict[str, int] = {n: 0 for n in nodes}
    for i in range(len(nodes)):
        changed = False
        for u, v in edges:
            if rank[v] < rank[u] + 1:
                rank[v] = rank[u] + 1
                changed = True
        if not changed:
            break
    # Ensure inputs remain minimal
    for n in inputs:
        rank[n] = min(rank.get(n, 0), 0)
    return rank


def layout(nodes: Set[str], edges: Set[Tuple[str, str]], inputs: Set[str]) -> Dict[str, Tuple[float, float]]:
    rank = rank_nodes(nodes, edges, inputs)
    # Group by rank
    by_rank: Dict[int, List[str]] = {}
    for n in nodes:
        by_rank.setdefault(rank[n], []).append(n)
    # Sort nodes within rank for determinism
    for k in by_rank:
        by_rank[k].sort()

    # Geometry
    x_spacing, y_spacing = 160, 80
    margin = 40
    pos: Dict[str, Tuple[float, float]] = {}
    for r in sorted(by_rank.keys()):
        col = by_rank[r]
        for i, n in enumerate(col):
            x = margin + r * x_spacing
            y = margin + i * y_spacing
            pos[n] = (x, y)
    return pos


def layout_by_time(nodes: Set[str], node_time: Dict[str, int]) -> Dict[str, Tuple[float, float]]:
    """Place nodes in vertical columns by their time index.
    Nodes with the same time share the same x; rows ordered alphabetically within time.
    Un-timed nodes (missing) will be placed at t=-1."""
    # Group nodes by time
    groups: Dict[int, List[str]] = {}
    for n in nodes:
        t = node_time.get(n, -1)
        groups.setdefault(t, []).append(n)
    for t in groups:
        groups[t].sort()

    # Geometry
    x_spacing, y_spacing = 180, 80
    margin = 40
    pos: Dict[str, Tuple[float, float]] = {}
    for col_idx, t in enumerate(sorted(groups.keys())):
        col = groups[t]
        for row_idx, n in enumerate(col):
            x = margin + col_idx * x_spacing
            y = margin + row_idx * y_spacing
            pos[n] = (x, y)
    return pos


def measure_node_label(label: str) -> Tuple[int, int]:
    # Simple width estimation based on label length
    w = max(60, 10 * len(label) + 20)
    h = 30
    return w, h


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )

def compute_edge_offsets(edges: Set[Tuple[str, str]], pos: Dict[str, Tuple[float, float]], labels: Optional[Dict[str, str]] = None) -> Dict[Tuple[str, str], Tuple[float, float]]:
    labels = labels or {}
    incoming: Dict[str, List[Tuple[float, str]]] = {}
    outgoing: Dict[str, List[Tuple[float, str]]] = {}
    for u, v in edges:
        uy = pos[u][1]
        vy = pos[v][1]
        incoming.setdefault(v, []).append((uy, u))
        outgoing.setdefault(u, []).append((vy, v))
    for v in incoming:
        incoming[v].sort(key=lambda t: (t[0], t[1]))
    for u in outgoing:
        outgoing[u].sort(key=lambda t: (t[0], t[1]))

    def offset_for(idx: int, count: int, step: float = 7.0) -> float:
        return (idx - (count - 1) / 2.0) * step

    incoming_index: Dict[Tuple[str, str], int] = {}
    for v, lst in incoming.items():
        for i, (_, u) in enumerate(lst):
            incoming_index[(u, v)] = i
    outgoing_index: Dict[Tuple[str, str], int] = {}
    for u, lst in outgoing.items():
        for i, (_, v) in enumerate(lst):
            outgoing_index[(u, v)] = i

    offsets: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for u, v in edges:
        si = outgoing_index.get((u, v), 0)
        so = offset_for(si, len(outgoing.get(u, [])))
        di = incoming_index.get((u, v), 0)
        do = offset_for(di, len(incoming.get(v, [])))
        offsets[(u, v)] = (so, do)
    return offsets


def render_svg(nodes: Set[str], edges: Set[Tuple[str, str]], pos: Dict[str, Tuple[float, float]], out_path: str, labels: Optional[Dict[str, str]] = None) -> None:
    # Compute canvas size
    labels = labels or {}
    max_x = max((pos[n][0] + measure_node_label(labels.get(n, n))[0] for n in nodes), default=0) + 40
    max_y = max((pos[n][1] + measure_node_label(labels.get(n, n))[1] for n in nodes), default=0) + 40
    parts: List[str] = []
    parts.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{int(max_x)}' height='{int(max_y)}'>")
    parts.append("<defs><marker id='arrow' markerWidth='10' markerHeight='7' refX='10' refY='3.5' orient='auto' markerUnits='strokeWidth'>" \
                 "<polygon points='0 0, 10 3.5, 0 7' fill='#333'/></marker></defs>")

    # Draw edges first
    offsets = compute_edge_offsets(edges, pos, labels)
    for u, v in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        w1, h1 = measure_node_label(labels.get(u, u))
        # start at right side of u, end at left side of v, with vertical offsets
        src_off, dst_off = offsets.get((u, v), (0.0, 0.0))
        sy = y1 + h1 / 2 + src_off
        ex, eh = x2, measure_node_label(labels.get(v, v))[1]
        ey = y2 + eh / 2 + dst_off
        # Quadratic Bézier curve for separation
        cx = (x1 + w1 + ex) / 2
        cy = (sy + ey) / 2
        parts.append(
            f"<path d='M {x1 + w1} {sy} Q {cx} {cy} {ex} {ey}' fill='none' stroke='#555' stroke-width='1.5' marker-end='url(#arrow)' />"
        )

    # Draw nodes
    for n in nodes:
        x, y = pos[n]
        label = labels.get(n, n)
        w, h = measure_node_label(label)
        parts.append(
            f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='6' ry='6' fill='#eef3ff' stroke='#4876f0' stroke-width='1.2' />"
        )
        parts.append(
            f"<text x='{x + w/2}' y='{y + h/2 + 4}' text-anchor='middle' font-family='monospace' font-size='12' fill='#1b2b57'>{svg_escape(label)}</text>"
        )

    parts.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def render_tk(nodes: Set[str], edges: Set[Tuple[str, str]], pos: Dict[str, Tuple[float, float]], labels: Optional[Dict[str, str]] = None) -> None:
    try:
        import tkinter as tk
    except Exception as e:
        print("Tkinter is not available for GUI display:", e, file=sys.stderr)
        return

    labels = labels or {}
    max_x = max((pos[n][0] + measure_node_label(labels.get(n, n))[0] for n in nodes), default=0) + 40
    max_y = max((pos[n][1] + measure_node_label(labels.get(n, n))[1] for n in nodes), default=0) + 40

    root = tk.Tk()
    root.title("Intrinsic Dependency Graph")
    canvas = tk.Canvas(root, width=max_x, height=max_y, bg="white")
    canvas.pack()

    def draw_arrow(x1, y1, x2, y2):
        canvas.create_line(x1, y1, x2, y2, fill="#555", width=1.5)
        # Simple arrowhead
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 8
        ax = x2
        ay = y2
        left = (ax - size * math.cos(angle - math.pi / 6), ay - size * math.sin(angle - math.pi / 6))
        right = (ax - size * math.cos(angle + math.pi / 6), ay - size * math.sin(angle + math.pi / 6))
        canvas.create_polygon(ax, ay, *left, *right, fill="#333")

    # Draw edges
    offsets = compute_edge_offsets(edges, pos, labels)
    for u, v in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        w1, h1 = measure_node_label(labels.get(u, u))
        w2, h2 = measure_node_label(labels.get(v, v))
        src_off, dst_off = offsets.get((u, v), (0.0, 0.0))
        sx, sy = x1 + w1, y1 + h1 / 2 + src_off
        ex, ey = x2, y2 + h2 / 2 + dst_off
        # Approximate curve with two segments
        cx = (sx + ex) / 2
        cy = (sy + ey) / 2
        canvas.create_line(sx, sy, cx, cy, fill="#555", width=1.5)
        draw_arrow(cx, cy, ex, ey)

    # Draw nodes
    for n in nodes:
        x, y = pos[n]
        label = labels.get(n, n)
        w, h = measure_node_label(label)
        canvas.create_rectangle(x, y, x + w, y + h, outline="#4876f0", fill="#eef3ff", width=1.2)
        canvas.create_text(x + w / 2, y + h / 2, text=label, font=("Courier", 10))

    root.mainloop()


def render_png(nodes: Set[str], edges: Set[Tuple[str, str]], pos: Dict[str, Tuple[float, float]], out_path: str, labels: Optional[Dict[str, str]] = None) -> None:
    labels = labels or {}
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as e:
        print("Pillow (PIL) is required for PNG output but is not available:", e, file=sys.stderr)
        print("Falling back to SVG output next to the requested PNG.", file=sys.stderr)
        alt = os.path.splitext(out_path)[0] + ".svg"
        render_svg(nodes, edges, pos, alt, labels=labels)
        return

    # Estimate canvas size
    max_x = max((pos[n][0] + measure_node_label(labels.get(n, n))[0] for n in nodes), default=0) + 40
    max_y = max((pos[n][1] + measure_node_label(labels.get(n, n))[1] for n in nodes), default=0) + 40
    img = Image.new("RGBA", (int(max_x), int(max_y)), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    def draw_arrow(x1, y1, x2, y2):
        draw.line((x1, y1, x2, y2), fill=(85, 85, 85), width=2)
        # Arrowhead
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 8
        left = (x2 - size * math.cos(angle - math.pi / 6), y2 - size * math.sin(angle - math.pi / 6))
        right = (x2 - size * math.cos(angle + math.pi / 6), y2 - size * math.sin(angle + math.pi / 6))
        draw.polygon([ (x2, y2), left, right ], fill=(51, 51, 51))

    # Draw edges
    offsets = compute_edge_offsets(edges, pos, labels)
    for u, v in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        w1, h1 = measure_node_label(labels.get(u, u))
        w2, h2 = measure_node_label(labels.get(v, v))
        src_off, dst_off = offsets.get((u, v), (0.0, 0.0))
        sx, sy = x1 + w1, y1 + h1 / 2 + src_off
        ex, ey = x2, y2 + h2 / 2 + dst_off
        # Quadratic-like polyline
        cx = (sx + ex) / 2
        cy = (sy + ey) / 2
        draw.line((sx, sy, cx, cy), fill=(85, 85, 85), width=2)
        draw_arrow(cx, cy, ex, ey)

    # Draw nodes
    for n in nodes:
        x, y = pos[n]
        label = labels.get(n, n)
        w, h = measure_node_label(label)
        # Box
        draw.rounded_rectangle([x, y, x + w, y + h], radius=6, outline=(72, 118, 240), width=2, fill=(238, 243, 255))
        # Text (centered approx)
        try:
            tw = draw.textlength(label, font=font)
        except Exception:
            tw = len(label) * 7
        th = 12
        draw.text((x + w / 2 - tw / 2, y + h / 2 - th / 2), label, fill=(27, 43, 87), font=font)

    img.save(out_path)


def dot_escape_label(s: str) -> str:
    return s.replace("\\", "\\\\").replace("\"", "\\\"")


def render_graphviz(nodes: Set[str], edges: Set[Tuple[str, str]], out_path: str, labels: Optional[Dict[str, str]] = None, node_time: Optional[Dict[str, int]] = None, align: str = "time", engine: str = "dot") -> None:
    labels = labels or {n: n for n in nodes}
    node_time = node_time or {}

    # Determine output format from extension
    ext = os.path.splitext(out_path)[1].lower()
    fmt = "png" if ext == ".png" else "svg"

    # Build DOT
    parts: List[str] = []
    parts.append("digraph G {")
    parts.append("  graph [rankdir=LR, splines=true, overlap=false, concentrate=false, nodesep=0.4, ranksep=0.6, fontname=\"monospace\"];")
    parts.append("  node [shape=box, style=\"rounded\", color=\"#4876f0\", penwidth=1.2, fontsize=10, fontname=\"monospace\"];")
    parts.append("  edge [color=\"#555\", penwidth=1.5, arrowhead=normal];")

    # Nodes
    for n in sorted(nodes):
        label = labels.get(n, n)
        parts.append(f'  "{n}" [label="{dot_escape_label(label)}"];')

    # Rank by time if requested
    if align == "time":
        groups: Dict[int, List[str]] = {}
        for n in nodes:
            t = node_time.get(n, -1)
            groups.setdefault(t, []).append(n)
        for t in sorted(groups.keys()):
            group = " ".join(f'"{n}"' for n in sorted(groups[t]))
            parts.append(f"  {{ rank=same; {group} }}")

    # Edges with optional minlen based on time distance
    for u, v in sorted(edges):
        attrs: List[str] = []
        if align == "time":
            tu = node_time.get(u)
            tv = node_time.get(v)
            if tu is not None and tv is not None:
                dt = max(1, (tv - tu))
                attrs.append(f"minlen={dt}")
        attr_str = (" [" + ", ".join(attrs) + "]") if attrs else ""
        parts.append(f'  "{u}" -> "{v}"{attr_str};')

    parts.append("}")
    dot_src = "\n".join(parts)

    # Call Graphviz
    cmd = [engine, f"-T{fmt}", "-o", out_path]
    completed = subprocess.run(cmd, input=dot_src.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        sys.stderr.write("[graphviz] dot failed, falling back to built-in renderer.\n")
        sys.stderr.write(completed.stderr.decode("utf-8", errors="ignore") + "\n")
        # Fallback to builtin SVG/PNG
        if fmt == "png":
            # Use time layout to keep alignment close
            pos = layout_by_time(nodes, node_time)
            render_png(nodes, edges, pos, out_path, labels=labels)
        else:
            pos = layout_by_time(nodes, node_time)
            render_svg(nodes, edges, pos, out_path, labels=labels)


def main():
    ap = argparse.ArgumentParser(description="Parse intrinsics-like code, build a variable dependency graph, and visualize it.")
    ap.add_argument("--input", "-i", help="Path to a file containing the inline assembly/intrinsics code. If omitted, use --sample.")
    ap.add_argument("--sample", action="store_true",  help="Use the built-in sample code (the provided COS_FLOAT32 snippet).")
    ap.add_argument("--out", "-o", default="graph.svg", help="Output path; .svg or .png")
    ap.add_argument("--gui", action="store_true", help="Open a Tkinter window to visualize the graph.")
    ap.add_argument("--no-self-loops", action="store_true", help="Remove self-loop edges (e.g., tmp -> tmp).")
    ap.add_argument("--dump-edges", action="store_true", help="Print edges to stdout.")
    ap.add_argument("--dump-op-edges", action="store_true", help="Print edges with op labels like 'u -> v (op)'.")
    ap.add_argument("--dump-with-time", action="store_true", help="Include [t=N] time index with text edge dumps.")
    ap.add_argument("--dump-nodes", action="store_true", help="Print nodes with their [t=N] time index.")
    ap.add_argument("--dump-liveness", action="store_true", help="Print alive register count per time.")
    ap.add_argument("--dump-liveness-regs", action="store_true", help="Also list alive registers per time.")
    ap.add_argument("--liveness-json", help="Write liveness JSON to path with per-time sets and intervals.")
    ap.add_argument("--include-params", action="store_true", help="Include parameters (no def time) in liveness.")
    ap.add_argument("--no-svg", action="store_true", help="Do not write SVG output; only dump text output.")
    ap.add_argument("--stage", choices=["raw", "ssa"], default="ssa", help="Graph stage: 'raw' (may include cycles) or 'ssa' (no cycles via versioning).")
    ap.add_argument("--align", choices=["rank", "time"], default="time", help="Layout: by dependency rank or by time index (statement order).")
    ap.add_argument("--backend", choices=["auto", "builtin", "graphviz"], default="auto", help="Which renderer to use: built-in or Graphviz (auto prefers Graphviz if available).")
    ap.add_argument("--graphviz-engine", default="dot", help="Graphviz engine to use (dot, neato, sfdp, ...).")
    ap.add_argument("--exclude", action="append", default=None, help="Variable name to exclude from edges. Can be repeated. Default: vl")
    ap.add_argument("--debug-dump", action="store_true", help="Print parser debug info (statements, params, nodes).")
    args = ap.parse_args()

    if not args.sample and not args.input:
        print("Either --input or --sample must be provided.", file=sys.stderr)
        sys.exit(2)

    text = SAMPLE_TEXT if args.sample else open(args.input, "r", encoding="utf-8").read()
    text = normalize_code(text)
    if args.debug_dump:
        print("[DEBUG] Normalized text:\n" + text)
    stmts = extract_statements(text)
    params = extract_params(text) or {"v", "vl"}  # fallback heuristic
    excludes = set(args.exclude) if args.exclude else {"vl"}
    nodes_raw, edges_raw, edges_with_ops_raw = build_edges(stmts, params, excludes=excludes)
    nodes_ssa, edges_ssa, edges_with_ops_ssa, node_time_ssa = build_edges_ssa(stmts, params, excludes=excludes)

    # Select stage graph
    if args.stage == "raw":
        nodes, edges, edges_with_ops = nodes_raw, edges_raw, edges_with_ops_raw
    else:
        nodes, edges, edges_with_ops = nodes_ssa, edges_ssa, edges_with_ops_ssa

    if args.no_self_loops:
        edges = {(u, v) for (u, v) in edges if u != v}
        edges_with_ops = [(u, v, op) for (u, v, op) in edges_with_ops if u != v]

    inputs = set(params)
    # Choose layout
    if args.align == "time":
        # Prefer SSA time map; fall back to raw approx.
        time_map: Dict[str, int] = {}
        if args.stage == "ssa":
            time_map = node_time_ssa
        else:
            var_first_time: Dict[str, int] = {}
            for idx, s in enumerate(stmts):
                s = s.strip()
                if not s:
                    continue
                if s.startswith("return "):
                    var_first_time["return"] = idx
                    continue
                m = re.search(r"(?:^|\\s)(?:auto|const)?\\s*([A-Za-z_]\\w*)\\s*=", s)
                if m:
                    var_first_time.setdefault(m.group(1), idx)
            time_map = var_first_time
        pos = layout_by_time(nodes, time_map)
    else:
        pos = layout(nodes, edges, inputs)

    # Build display labels with time index
    labels: Dict[str, str] = {}
    if args.stage == "ssa":
        # time = statement order; params not assigned in body -> time -1
        for n in nodes:
            t = node_time_ssa.get(n, -1)
            labels[n] = f"{n} [t={t}]"
    else:
        # Raw mode: approximate time as first assignment index; params=-1
        var_first_time: Dict[str, int] = {}
        for idx, s in enumerate(stmts):
            s = s.strip()
            if not s:
                continue
            if s.startswith("return "):
                var_first_time["return"] = idx
                continue
            m = re.search(r"(?:^|\\s)(?:auto|const)?\\s*([A-Za-z_]\\w*)\\s*=", s)
            if m:
                var_first_time.setdefault(m.group(1), idx)
        for n in nodes:
            t = var_first_time.get(n, -1)
            labels[n] = f"{n} [t={t}]"

    # Write output using selected backend
    out_ext = os.path.splitext(args.out)[1].lower()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Decide backend
    use_graphviz = False
    if args.backend == "graphviz":
        use_graphviz = True
    elif args.backend == "auto" and shutil.which(args.graphviz_engine):
        use_graphviz = True

    if use_graphviz:
        # Build time map
        if args.stage == "ssa":
            time_map = dict(node_time_ssa)
        else:
            var_first_time: Dict[str, int] = {}
            for idx, s in enumerate(stmts):
                s = s.strip()
                if not s:
                    continue
                if s.startswith("return "):
                    var_first_time["return"] = idx
                    continue
                m = re.search(r"(?:^|\\s)(?:auto|const)?\\s*([A-Za-z_]\\w*)\\s*=", s)
                if m:
                    var_first_time.setdefault(m.group(1), idx)
            time_map = var_first_time
        render_graphviz(nodes, edges, args.out, labels=labels, node_time=time_map, align=args.align, engine=args.graphviz_engine)
    else:
        # Built-in backends as before
        if out_ext == ".png":
            render_png(nodes, edges, pos, args.out, labels=labels)
        elif not args.no_svg:
            render_svg(nodes, edges, pos, args.out, labels=labels)

    # Build display labels with time index
    labels: Dict[str, str] = {}
    if args.stage == "ssa":
        # time = statement order; params not assigned in body -> time -1
        for n in nodes:
            t = node_time_ssa.get(n, -1)
            labels[n] = f"{n} [t={t}]"
    else:
        # Raw mode: approximate time as first assignment index; params=-1
        var_first_time: Dict[str, int] = {}
        for idx, s in enumerate(stmts):
            s = s.strip()
            if not s:
                continue
            if s.startswith("return "):
                var_first_time["return"] = idx
                continue
            m = re.search(r"(?:^|\\s)(?:auto|const)?\\s*([A-Za-z_]\\w*)\\s*=", s)
            if m:
                var_first_time.setdefault(m.group(1), idx)
        for n in nodes:
            t = var_first_time.get(n, -1)
            labels[n] = f"{n} [t={t}]"

    # Common time map for dumps
    if args.stage == "ssa":
        time_map_print: Dict[str, int] = dict(node_time_ssa)
    else:
        time_map_print = {}
        for idx, s in enumerate(stmts):
            s = s.strip()
            if not s:
                continue
            if s.startswith("return "):
                time_map_print["return"] = idx
                continue
            m = re.search(r"(?:^|\\s)(?:auto|const)?\\s*([A-Za-z_]\\w*)\\s*=", s)
            if m:
                time_map_print.setdefault(m.group(1), idx)

    if args.debug_dump:
        print("[DEBUG] Params:", sorted(params))
        print("[DEBUG] Statements:", len(stmts))
        for i, s in enumerate(stmts[:10]):
            print(f"  [{i}] {s}")
        print("[DEBUG] Nodes:", sorted(nodes))
        print("[DEBUG] Edges:", len(edges))
    if args.dump_nodes:
        for n in sorted(nodes, key=lambda x: (time_map_print.get(x, -1), x)):
            print(f"[t={time_map_print.get(n, -1)}] {n}")
    if args.dump_edges:
        for u, v in sorted(edges):
            if args.dump_with_time:
                print(f"[t={time_map_print.get(u, -1)}] {u} -> [t={time_map_print.get(v, -1)}] {v}")
            else:
                print(f"{u} -> {v}")
    if args.dump_op_edges:
        for u, v, op in sorted(edges_with_ops, key=lambda t: (t[0], t[1], t[2] or '')):
            label = f" ({op})" if op else ""
            if args.dump_with_time:
                print(f"[t={time_map_print.get(u, -1)}] {u} -> [t={time_map_print.get(v, -1)}] {v}{label}")
            else:
                print(f"{u} -> {v}{label}")

    # Optional: liveness (SSA preferred)
    intervals: Dict[str, Tuple[int, int]] = {}
    alive_by_time: Dict[int, List[str]] = {}
    if args.stage == "ssa":
        intervals, alive_by_time = compute_liveness_ssa(nodes_ssa, edges_ssa, node_time_ssa, include_params=args.include_params)
    else:
        # Best effort: compute using selected nodes/edges and approximated times
        # Build a time map similar to labels above
        var_first_time: Dict[str, int] = {}
        for idx, s in enumerate(stmts):
            s = s.strip()
            if not s:
                continue
            if s.startswith("return "):
                var_first_time["return"] = idx
                continue
            m = re.search(r"(?:^|\\s)(?:auto|const)?\\s*([A-Za-z_]\\w*)\\s*=", s)
            if m:
                var_first_time.setdefault(m.group(1), idx)
        intervals, alive_by_time = compute_liveness_ssa(nodes, edges, var_first_time, include_params=args.include_params)

    # Dumps for liveness
    if args.dump_liveness or args.dump_liveness_regs or args.liveness_json:
        # Choose timespan to report in order
        times = sorted(alive_by_time.keys())
        if args.dump_liveness or args.dump_liveness_regs:
            for t in times:
                regs = alive_by_time[t]
                if args.dump_liveness_regs:
                    print(f"t={t}: {len(regs)} -> {', '.join(regs)}")
                else:
                    print(f"t={t}: {len(regs)}")
        if args.liveness_json:
            data = {
                "alive_by_time": {str(t): regs for t, regs in alive_by_time.items()},
                "intervals": {k: [s, e] for k, (s, e) in intervals.items()},
            }
            os.makedirs(os.path.dirname(args.liveness_json) or ".", exist_ok=True)
            with open(args.liveness_json, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    if args.gui:
        render_tk(nodes, edges, pos, labels=labels)
    else:
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
