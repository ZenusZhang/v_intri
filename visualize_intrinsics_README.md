# visualize_intrinsics.py — RVV Intrinsics Graph + Liveness

This script parses a RISC‑V Vector (RVV) intrinsic kernel written in C/C++ (typically macro‑expanded inline code), builds a variable dependency graph, performs SSA versioning, annotates nodes with a sequential "time" index (statement order), and renders the graph to SVG or PNG. It can also dump text forms of the graph, and compute/register liveness (vector register pressure) over time.

## Features

- Parses intrinsics‑style code (after lightweight macro normalization) to extract statements, variables, and data dependencies.
- SSA mode: each assignment to a variable creates a new SSA node (e.g., `y_f3`). Eliminates cycles and makes liveness analysis straightforward.
- Time index: every statement is assigned an increasing integer time `t` starting at 0; nodes carry `[t=N]` for their definition time; `return` also gets a time.
- Rendering backends:
  - Built‑in SVG and PNG renderers with time‑aligned layout (`--align time`) and curved edges that avoid overlap.
  - Optional Graphviz backend (`--backend graphviz` or `--backend auto`) for spline routing and automatic layout, grouped by time.
- Text dumps: edges (optionally with op labels and time) and node lists with time.
- Liveness analysis (SSA): per‑time alive set and intervals, focusing on vector registers. Scalars prefixed with `s_` are ignored in liveness by default.
- JSON export: liveness info (alive‑by‑time and intervals) as a machine‑readable artifact.

## Requirements

- Python 3.8+.
- Optional for PNG output: Pillow (`pip install pillow`). If missing, script falls back to SVG.
- Optional for GUI preview: Tkinter (usually available on many Python builds).
- Optional for Graphviz backend: Graphviz binaries (e.g., `dot`). When available, `--backend auto` uses it.

## Usage

Basic (use the built‑in sample):

```bash
python3 tools/visualize_intrinsics.py --sample --stage ssa --align time --out graph.svg
```

From a file:

```bash
python3 tools/visualize_intrinsics.py --input path/to/code.cpp --stage ssa --align time --out graph.png
```

Key options:

- Input and stage
  - `--input/-i FILE`: source code to parse (or use `--sample`).
  - `--stage raw|ssa`: raw graph or SSA graph (default: `ssa`).
- Layout
  - `--align rank|time`: layout by dependency rank or by statement time (default: `time`).
- Backends and output
  - `--backend auto|builtin|graphviz` (default: `auto`).
  - `--graphviz-engine dot|neato|sfdp` (default: `dot`).
  - `--out/-o PATH`: output file. `.svg` or `.png` decide format. Use `--gui` to open a Tk window instead/as well.
  - `--no-svg`: skip SVG write (useful when only dumping text).
- Graph content controls
  - `--exclude NAME` (repeatable): drop variable(s) from edges (default excludes `vl`).
- Text dumps
  - `--dump-nodes`: list nodes with `[t=N]`.
  - `--dump-edges`: list edges; add `--dump-with-time` to include source/target times.
  - `--dump-op-edges`: edges with an inferred intrinsic op label.
- Liveness (SSA preferred)
  - `--dump-liveness`: print count of alive vector regs at each time.
  - `--dump-liveness-regs`: also list the alive SSA registers.
  - `--liveness-json FILE`: write JSON with `alive_by_time` and `intervals`.
  - `--include-params`: include parameters (no def time) in liveness (off by default).

Examples:

```bash
# SSA edges with time, no file output
python3 tools/visualize_intrinsics.py \
  --input code.cpp --stage ssa --dump-op-edges --dump-with-time --no-svg

# Time‑aligned PNG with Graphviz backend (if installed)
python3 tools/visualize_intrinsics.py \
  --input code.cpp --stage ssa --align time --backend graphviz --out graph.png

# Liveness: counts and registers per time, plus JSON export
python3 tools/visualize_intrinsics.py \
  --input code.cpp --stage ssa --dump-liveness --dump-liveness-regs \
  --liveness-json liveness.json --no-svg
```

## Time Semantics

- The script assigns a monotonic time `t` to each statement in order of appearance, starting at 0.
- In SSA mode, each SSA node’s label includes its definition time: `name_fK [t=N]`.
- The `return` node is also time‑stamped, allowing edges into `return` to carry a use time.

## Liveness (Vector Registers)

- SSA liveness is computed on the SSA graph:
  - Interval for node `u`: `[def_time(u), max_use_time(u)]` (inclusive). If no uses, interval collapses to `def_time(u)`.
  - Per‑time alive set: all `u` such that `def_time(u) ≤ t ≤ max_use_time(u)`.
- Vector vs Scalar:
- Variables prefixed with `s_` are considered scalar and are ignored in liveness (they do not count toward alive vector registers).
  - If a vector feeds an `s_*` node, that still contributes a use and may extend the vector’s live interval (the use time is the sink time).
- Parameters are excluded by default; enable `--include-params` to include them.
- In `raw` stage, liveness is approximate (first assignment times are used). Prefer `ssa` for correctness.

## Rendering Notes

- Built‑in renderer:
  - `--align time` places all nodes with the same `t` in one column.
  - Curved edges and small per‑edge vertical offsets reduce overlap when multiple edges share the same endpoints or rows.
- Graphviz backend:
  - Uses `rankdir=LR` and groups nodes by time with `rank=same` for time‑aligned columns.
  - Spline edges and `minlen` based on time distance improve clarity and reduce crossings.
  - Choose engines (`dot`, `neato`, `sfdp`) via `--graphviz-engine`.

## JSON Schema

When `--liveness-json FILE` is provided, the script writes:

```json
{
  "alive_by_time": {
    "0": ["r_f1", "y_f1"],
    "1": ["r_f1", "y_f1", "r2_f1"],
    "2": ["y_f2"]
  },
  "intervals": {
    "r_f1": [0, 3],
    "y_f1": [0, 2],
    "y_f2": [2, 5]
  }
}
```

Times and register names above are illustrative.

## Limitations & Tips

- Parsing is heuristic and assumes C/C++‑like intrinsics code after simple macro normalization. It will not fully parse arbitrary C++.
- If your code uses unusual macros or token‑pasting, consider pre‑expanding it or adjusting `normalize_code`.
- Use `--exclude NAME` to drop non‑dataflow arguments (e.g., `vl`) from edges.
- Prefer `--stage ssa` for cleaner graphs and correct liveness.
- If PNG rendering fails due to missing Pillow, the script falls back to SVG automatically.
- If Graphviz is unavailable or `dot` errors, the script falls back to the built‑in renderer.

## Troubleshooting

- No edges drawn or too few: check `--exclude` settings and verify your variable names are recognized as candidates.
- Overlapping edges in built‑in renderer: ensure you’re on time‑aligned layout; Graphviz usually handles routing better.
- GUI errors: your Python may lack Tk; use `--out graph.svg` or install Tkinter.

---

For questions or improvements, feel free to open an issue or suggest enhancements (e.g., hiding scalar nodes in the graph, CSV dumps, or peak pressure annotations on the timeline).
