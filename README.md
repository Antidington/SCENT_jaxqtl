# pySCENT

SCENT reimplemented in Python with JAX-powered acceleration for high-performance computing.

---

## 1. SCENT Framework Introduction

SCENT is a statistical framework for mapping enhancer-gene regulatory relationships using single-cell multi-omics data that integrates:
- **RNA-seq**: Gene expression levels
- **ATAC-seq**: Chromatin accessibility at regulatory regions

### Core Methodology

SCENT uses **Generalized Linear Models (GLMs)** to model the relationship between gene expression and chromatin accessibility:

```
Gene Expression ~ Intercept + Covariates + Peak Accessibility
```

**Key components:**
- **Response**: RNA-seq counts (Poisson or Negative Binomial distribution)
- **Predictor**: Binarized ATAC-seq signal (1 if accessible, 0 if not)
- **Covariates**: Batch effects, sequencing depth, etc.
- **Inference**: Bootstrap testing for robust p-value estimation

SCENT identifies significant peak-gene associations in a cell-type specific manner, accounting for technical confounders and biological variability.

**Reference:**
- Sakaue et al. ["Tissue-specific enhancer-gene maps from multimodal single-cell data identify causal disease alleles"](https://www.nature.com/articles/s41588-024-01682-1) *Nature Genetics* (2024)
- Original R implementation: [SCENT on GitHub](https://github.com/immunogenomics/SCENT)

---

## 2. pySCENT Features

pySCENT reimplements SCENT in Python using JAX, providing substantial performance improvements while maintaining statistical consistency with the original implementation.

### Key Features

* **GPU/TPU Acceleration**: JAX-based computation enables hardware acceleration
* **Multi-GPU Support**: Automatically shards gene-peak pairs across multiple GPUs
* **High Consistency**: Validated against original SCENT
* **Adaptive Bootstrap Strategy**: Intelligent sampling for efficiency
* **Vectorized Parallelization**: `jax.vmap` for simultaneous bootstrap execution

---

## 3. Installation

### Install from Source

Using `uv` (recommended):
```bash
git clone https://github.com/Antidington/pySCENT.git
cd pySCENT
uv sync
```

Using `pip`:
```bash
git clone https://github.com/Antidington/pySCENT.git
cd pySCENT
pip install -e .
```

### Dependencies

Core dependencies (from `pyproject.toml`):
- `jaxqtl` - JAX-based QTL mapping library
- `lineax` ≥ 0.0.8 - Linear algebra operations
- `qtl` ≥ 0.1.10 - QTL utilities

### CPU vs GPU/TPU

pySCENT runs on **CPU by default** — no extra configuration needed. To enable GPU or TPU acceleration, install the corresponding JAX build:

**NVIDIA GPU (CUDA 12):**
```bash
# uv
uv pip install -U "jax[cuda12]"

# pip
pip install -U "jax[cuda12]"
```

**Apple Silicon (Metal):**
```bash
pip install jax-metal
```

**Google Cloud TPU:**
```bash
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Verify which backend is active:
```python
import jax
print(jax.default_backend())  # "cpu", "gpu", or "tpu"
print(jax.devices())          # list available devices
```

pySCENT does not require any code changes to switch backends — JAX automatically dispatches to the available accelerator. For more details, see the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

### GPU Memory Control

By default JAX pre-allocates 75% of GPU memory. For large datasets set these environment variables **before importing pySCENT**:

```bash
# Disable preallocation (allocate on demand)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Or cap the fraction (e.g. 40%)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
```

---

## 4. Usage

### Quick Start

```python
import jax.random as random
from pyscent import io

# Create SCENT object from input files
scent_obj = io.create_scent_object(
    rna_matrix="data/rna_matrix.csv",      # Gene × Cell expression matrix
    atac_matrix="data/atac_matrix.csv",    # Peak × Cell accessibility matrix
    meta_data="data/metadata.csv",         # Cell metadata
    peak_info="data/peak_info.csv",        # Gene-peak pairs to test
    covariates=["batch", "n_counts"],      # Covariates to adjust for
    celltype_col="cell_type"               # Column name for cell type
)

# CPU (default) — control thread count with ncores
results = scent_obj.run_scent(
    celltype="T_cell",
    regr="poisson",
    bootstrap_samples=100,
    min_nonzero_frac=0.05,
    ncores=4,
    key=random.PRNGKey(42),
)

# Single GPU — prints device info, runs on GPU 0
results = scent_obj.run_scent(
    celltype="T_cell",
    gpu_devices=[0],
)

# Multi-GPU — automatically shards gene-peak pairs across GPUs 0 and 1
results = scent_obj.run_scent(
    celltype="T_cell",
    gpu_devices=[0, 1],
)

# Save results
io.write_results(results, "output/scent_results.csv")

# Display top associations
for res in sorted(results, key=lambda x: x.boot_basic_p)[:5]:
    print(f"Gene: {res.gene}, Peak: {res.peak}, "
          f"Beta: {res.beta:.3f}, P-value: {res.boot_basic_p:.2e}")
```

### GPU Device Selection

When a GPU backend is used, pySCENT prints the detected devices to stdout:

```
Detected 3 GPU device(s):
  [0] NVIDIA A100-SXM4-80GB | VRAM: 81920 MiB | Load: 12%
  [1] NVIDIA A100-SXM4-80GB | VRAM: 81920 MiB | Load:  0%
  [2] NVIDIA A100-SXM4-80GB | VRAM: 81920 MiB | Load:  0%
Using GPU [0]
```

| `gpu_devices` | Behaviour |
|---|---|
| `None` (default) | Falls back to `device` argument (`"auto"` → GPU 0 if available) |
| `[0]` | Single-GPU mode on GPU 0 |
| `[0, 1, 2]` | Multi-GPU mode — pairs sharded round-robin, one subprocess per GPU |

### Input Data Format

RNA and ATAC matrices are stored internally as sparse matrices (scipy CSR, analogous to R's `dgCMatrix`). The following input formats are supported:

| Format | Extension | Row/Column Names |
|--------|-----------|-----------------|
| CSV | `.csv` | First column = row names, header = column names |
| TSV | `.tsv` | Same as CSV, tab-separated |
| H5AD | `.h5ad` | `var_names` = row names, `obs_names` = column names (auto-transposed from cells×genes to genes×cells) |
| MTX | `.mtx`, `.mtx.gz` | Auto-loaded from companion `{stem}_genes.tsv` / `{stem}_features.tsv` and `{stem}_barcodes.tsv` files if present alongside the MTX |

**RNA matrix** — genes × cells, raw counts (no normalization):
```
        Cell1  Cell2  Cell3  ...
Gene1   10     5      8      ...
Gene2   0      3      12     ...
```

**ATAC matrix** — peaks × cells, raw counts:
```
                          Cell1  Cell2  Cell3  ...
chr1:1000-2000            5      0      3      ...
chr2:5000-6000            8      10     2      ...
```

**Metadata** — one row per cell, must contain a `cell` (or `cell_id`) column:
```
cell_id,cell_type,batch,n_counts
Cell1,T_cell,batch1,5000
Cell2,B_cell,batch1,4500
```

**Peak info** — gene-peak pairs to test (first two columns used):
```
gene,peak
Gene1,chr1:1000-2000
Gene2,chr2:5000-6000
```

### Output Format

Results are saved as CSV with columns:
- `gene`: Gene name
- `peak`: Peak coordinates
- `beta`: Regression coefficient (effect size)
- `se`: Standard error
- `z`: Z-score
- `p`: Wald test p-value
- `boot_basic_p`: Bootstrap p-value (recommended)

---
