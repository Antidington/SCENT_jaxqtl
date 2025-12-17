# SCENT_jaxQTL

A GPU-accelerated Python implementation of SCENT (Single-Cell ENhancer Target) for identifying cell-type specific enhancer-gene regulatory links from single-cell multi-omics data.

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

## 2. SCENT_jaxQTL Features

SCENT_jaxQTL reimplements SCENT in Python using JAX, providing substantial performance improvements while maintaining statistical consistency with the original implementation.

### Key Features

**GPU/TPU Acceleration**: JAX-based computation enables hardware acceleration
**High Consistency**: Validated against original SCENT
**Adaptive Bootstrap Strategy**: Intelligent sampling for efficiency
**Vectorized Parallelization**: `jax.vmap` for simultaneous bootstrap execution

---

## 3. Installation

### Prerequisites
- Python ≥ 3.10
- JAX (GPU support optional but recommended)

### Install from Source

Using `uv` (recommended):
```bash
# Clone repository
git clone https://github.com/Antidington/SCENT_jaxqtl.git
cd SCENT_jaxqtl

# Install with uv
uv sync
```

Using `pip`:
```bash
# Clone repository
git clone https://github.com/Antidington/SCENT_jaxqtl.git
cd SCENT_jaxqtl

# Install in editable mode
pip install -e .
```

### Dependencies

Core dependencies (from `pyproject.toml`):
- `jaxqtl` - JAX-based QTL mapping library
- `lineax` ≥ 0.0.8 - Linear algebra operations
- `notebook` ≥ 7.4.6 - Jupyter notebook support
- `qtl` ≥ 0.1.10 - QTL utilities

JAX will be installed automatically. For GPU support, follow [JAX installation guide](https://github.com/google/jax#installation).

---

## 4. Usage

### Quick Start

```python
import jax.random as random
from SCENT_jaxqtl import io

# Create SCENT object from input files
scent_obj = io.create_scent_object(
    rna_matrix="data/rna_matrix.csv",      # Gene × Cell expression matrix
    atac_matrix="data/atac_matrix.csv",    # Peak × Cell accessibility matrix
    meta_data="data/metadata.csv",         # Cell metadata
    peak_info="data/peak_info.csv",        # Gene-peak pairs to test
    covariates=["batch", "n_counts"],      # Covariates to adjust for
    celltype_col="cell_type"               # Column name for cell type
)

# Run SCENT analysis for specific cell type
results = scent_obj.run_scent(
    celltype="T_cell",                     # Cell type to analyze
    regr="poisson",                        # Regression model: "poisson" or "negbin"
    bootstrap_samples=100,                 # Initial bootstrap samples
    key=random.PRNGKey(42)                 # Random seed for reproducibility
)

# Save results
io.write_results(results, "output/scent_results.csv")

# Display top associations
for res in sorted(results, key=lambda x: x.boot_basic_p)[:5]:
    print(f"Gene: {res.gene}, Peak: {res.peak}, "
          f"Beta: {res.beta:.3f}, P-value: {res.boot_basic_p:.2e}")
```

### Input Data Format

**RNA matrix** (`rna_matrix.csv`):
```
        Cell1  Cell2  Cell3  ...
Gene1   10     5      8      ...
Gene2   0      3      12     ...
```

**ATAC matrix** (`atac_matrix.csv`):
```
                          Cell1  Cell2  Cell3  ...
chr1:1000-2000            5      0      3      ...
chr2:5000-6000            8      10     2      ...
```

**Metadata** (`metadata.csv`):
```
cell_id,cell_type,batch,n_counts
Cell1,T_cell,batch1,5000
Cell2,B_cell,batch1,4500
```

**Peak info** (`peak_info.csv`):
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




