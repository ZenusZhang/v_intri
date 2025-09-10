!!! This is a repo completely built by codex and claude code, python file is produced by codex, readme and building repo is done by Claude Code

# V_intri - RVV Intrinsics Visualizer

A powerful tool for visualizing RISC-V Vector (RVV) intrinsic kernels with dependency graphs and liveness analysis.

## Overview

This tool parses RVV intrinsic code written in C/C++, builds variable dependency graphs, performs SSA versioning, and provides comprehensive visualization and analysis capabilities including:

- **Dependency Graph Visualization**: Interactive SVG graphs showing data flow between variables
- **SSA Versioning**: Automatic static single assignment conversion for cleaner analysis
- **Time-indexed Analysis**: Each statement is assigned a sequential time index for temporal analysis
- **Liveness Analysis**: Track vector register pressure over time with detailed interval analysis
- **Multiple Output Formats**: SVG, PNG, and JSON export capabilities

## Features

### 🔍 Code Analysis
- Parses RVV intrinsics-style code with lightweight macro normalization
- Extracts statements, variables, and data dependencies automatically
- Supports both raw and SSA (Static Single Assignment) analysis modes

### 📊 Visualization
- **Built-in SVG/PNG renderer**: Time-aligned layout with curved edges
- **Graphviz integration**: Automatic layout with spline routing
- **Interactive GUI preview**: Tkinter-based viewer for immediate feedback
- **Time-aligned layout**: Nodes grouped by statement execution order

### 📈 Liveness Analysis
- **Vector register tracking**: Monitor register pressure over time
- **Interval analysis**: Track live ranges for each variable
- **JSON export**: Machine-readable liveness data for integration
- **Scalar separation**: Automatic filtering of scalar variables (prefixed with `s_`)

## Quick Start

### Installation
```bash
git clone https://github.com/Rubiczhang/v_intri.git
cd v_intri
```

### Basic Usage
Run with the built-in sample to see immediate results:
```bash
python3 visualize_intrinsics.py --sample
```

This will generate `graph.svg` showing the dependency graph of the sample kernel.

### Sample Results

The sample kernel implements a cosine function using range reduction and polynomial approximation with the following characteristics:

#### Nodes (Time-indexed SSA Variables)
```
Parameters (t=-1): half, n, ni, odd, r, r2, s_c3, s_c5, s_c7, s_c9, tmp, v, vl, y

Computation:
t=0:  n_f1
t=1:  half_f1
t=2:  r_f1
t=3:  n_f2
t=4:  ni_f1
t=5:  n_f3
t=6:  odd_f1
t=7:  n_f4
t=8:  odd_f2
t=9:  r_f2
t=10: r_f3
t=11: r_f4
t=12: r2_f1
t=13: s_c9_f1
t=14: s_c7_f1
t=15: s_c5_f1
t=16: s_c3_f1
t=17: y_f1
t=18: y_f2
t=19: y_f3
t=20: y_f4
t=21: y_f5
t=22: y_f6
t=23: y_f7
t=24: y_f8
t=25: tmp_f1
t=26: tmp_f2
t=27: return
```

#### Dependency Graph Edges
```
half_f1 -> n_f2
n_f1 -> n_f2
n_f2 -> ni_f1
n_f3 -> n_f4
n_f4 -> r_f2
n_f4 -> r_f3
n_f4 -> r_f4
ni_f1 -> n_f3
ni_f1 -> odd_f1
odd_f1 -> odd_f2
odd_f2 -> tmp_f2
r2_f1 -> y_f1
r2_f1 -> y_f3
r2_f1 -> y_f5
r2_f1 -> y_f8
r_f1 -> n_f2
r_f1 -> r_f2
r_f2 -> r_f3
r_f3 -> r_f4
r_f4 -> r2_f1
r_f4 -> y_f7
r_f4 -> y_f8
s_c3_f1 -> y_f6
s_c5_f1 -> y_f4
s_c7_f1 -> y_f2
s_c9_f1 -> y_f1
tmp_f1 -> tmp_f2
tmp_f2 -> return
v -> r_f1
y_f1 -> y_f2
y_f2 -> y_f3
y_f3 -> y_f4
y_f4 -> y_f5
y_f5 -> y_f6
y_f6 -> y_f7
y_f7 -> y_f8
y_f8 -> tmp_f1
```

#### Liveness Analysis (Register Pressure)
```
t=0:  1 alive  -> n_f1
t=1:  2 alive  -> half_f1, n_f1
t=2:  3 alive  -> half_f1, n_f1, r_f1
t=3:  4 alive  -> half_f1, n_f1, n_f2, r_f1
t=4:  3 alive  -> n_f2, ni_f1, r_f1
t=5:  3 alive  -> n_f3, ni_f1, r_f1
t=6:  4 alive  -> n_f3, ni_f1, odd_f1, r_f1
t=7:  4 alive  -> n_f3, n_f4, odd_f1, r_f1
t=8:  4 alive  -> n_f4, odd_f1, odd_f2, r_f1
t=9:  4 alive  -> n_f4, odd_f2, r_f1, r_f2
t=10: 4 alive  -> n_f4, odd_f2, r_f2, r_f3
t=11: 4 alive  -> n_f4, odd_f2, r_f3, r_f4
t=12: 3 alive  -> odd_f2, r2_f1, r_f4
t=13: 3 alive  -> odd_f2, r2_f1, r_f4
t=14: 3 alive  -> odd_f2, r2_f1, r_f4
t=15: 3 alive  -> odd_f2, r2_f1, r_f4
t=16: 3 alive  -> odd_f2, r2_f1, r_f4
t=17: 4 alive  -> odd_f2, r2_f1, r_f4, y_f1
t=18: 5 alive  -> odd_f2, r2_f1, r_f4, y_f1, y_f2
t=19: 5 alive  -> odd_f2, r2_f1, r_f4, y_f2, y_f3
t=20: 5 alive  -> odd_f2, r2_f1, r_f4, y_f3, y_f4
t=21: 5 alive  -> odd_f2, r2_f1, r_f4, y_f4, y_f5
t=22: 5 alive  -> odd_f2, r2_f1, r_f4, y_f5, y_f6
t=23: 5 alive  -> odd_f2, r2_f1, r_f4, y_f6, y_f7
t=24: 5 alive  -> odd_f2, r2_f1, r_f4, y_f7, y_f8
t=25: 3 alive  -> odd_f2, tmp_f1, y_f8
t=26: 3 alive  -> odd_f2, tmp_f1, tmp_f2
t=27: 2 alive  -> return, tmp_f2
```

**Key Observations:**
- **Peak register pressure**: 5 registers sustained from t=17 to t=24 during polynomial evaluation
- **Total execution time**: 29 time steps (t=-1 to t=27)
- **Efficient register usage**: Steady register pressure during the main computation phase
- **Clear computation phases**: Range reduction (t=0-12), polynomial setup (t=13-16), evaluation (t=17-24), and finalization (t=25-27)
- **Scalar constants**: Note that s_c3, s_c5, s_c7, s_c9 are prefixed with `s_` indicating they are treated as scalar constants

### Sample Visualization

Here's the generated dependency graph from the sample kernel:

![Sample RVV Dependency Graph](graph.svg)

This graph shows:
- **Nodes**: Each SSA variable with its time index (`[t=N]`)
- **Edges**: Data dependencies between variables
- **Layout**: Time-aligned from left to right (t=0 to t=28)
- **Colors**: Different node types (parameters, computations, return)
- **Curved paths**: Minimize edge overlap for better readability

## Advanced Usage

### Analyze Your Own Code
```bash
python3 visualize_intrinsics.py --input your_kernel.cpp --stage ssa --align time --out your_graph.svg
```

### Export Liveness Data
```bash
python3 visualize_intrinsics.py --input your_kernel.cpp --dump-liveness --liveness-json liveness.json
```

### Graphviz Backend (Higher Quality)
```bash
python3 visualize_intrinsics.py --input your_kernel.cpp --backend graphviz --out high_quality_graph.png
```

## Requirements

- Python 3.8+
- Optional dependencies:
  - Pillow (`pip install pillow`) for PNG output
  - Tkinter (usually included) for GUI preview
  - Graphviz binaries for enhanced rendering

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input/-i FILE` | Input source file | Use `--sample` |
| `--sample` | Use built-in sample | Off |
| `--stage raw|ssa` | Analysis stage | `ssa` |
| `--align rank|time` | Layout method | `time` |
| `--backend auto|builtin|graphviz` | Rendering backend | `auto` |
| `--out/-o PATH` | Output file path | `graph.svg` |
| `--dump-nodes` | List nodes with time | Off |
| `--dump-edges` | List edges | Off |
| `--dump-liveness` | Show register pressure | Off |
| `--liveness-json FILE` | Export liveness to JSON | Off |

## Applications

- **Compiler Development**: Understand register pressure in RVV kernels
- **Performance Analysis**: Identify optimization opportunities in vector code
- **Educational**: Visualize data dependencies in vector computations
- **Research**: Analyze and compare different RVV coding patterns

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

For more detailed information about the implementation and advanced features, see the [technical documentation](https://github.com/Rubiczhang/v_intri/wiki).
