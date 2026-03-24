# -*- coding: utf-8 -*-
# @Time       : 2025/9/26
# @Author     : Jichen Wen
# E-mail      : wenjichen.big@gmail.com / wenjichen@cncb.ac.cn
# @Description: Core SCENT functionality using jaxqtl

from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as rdm
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from scipy import sparse

from jaxqtl.families.distribution import NegativeBinomial, Poisson
from jaxqtl.infer.glm import GLM
from jaxqtl.infer.stderr import FisherInfoError

from .bootstrap import bootstrap_test


def _resolve_device(device: str) -> jax.Device:
    """Description:
        Resolve a device string to a concrete ``jax.Device`` instance.

        ``"auto"`` probes backends in order gpu → tpu → cpu and returns the
        first available device.

    Args:
        device: One of ``"auto"``, ``"cpu"``, ``"gpu"``, or ``"tpu"``.

    Returns:
        A ``jax.Device`` object for the selected backend.
    """
    if device == "auto":
        for backend in ("gpu", "tpu", "cpu"):
            try:
                devices = jax.devices(backend)
                if devices:
                    return devices[0]
            except RuntimeError:
                continue
        return jax.devices("cpu")[0]

    try:
        devices = jax.devices(device)
    except RuntimeError:
        raise ValueError(f"JAX backend '{device}' is not available. Install the corresponding JAX build.")
    if not devices:
        raise ValueError(f"No {device} device found.")
    return devices[0]


class SCENTResult(NamedTuple):
    """Results from SCENT analysis"""

    gene: str
    peak: str
    beta: float
    se: float
    z: float
    p: float
    boot_basic_p: float


class SCENTObject(eqx.Module):
    """SCENT object for single-cell epigenome and transcriptome analysis.

    Mirrors the original R SCENT S4 class: matrices are stored as sparse
    (scipy CSR, analogous to R dgCMatrix) while row/column names are kept
    in separate lists (analogous to Dimnames slots).
    """

    rna: Any  # Sparse CSR matrix (genes x cells)
    atac: Any  # Sparse CSR matrix (peaks x cells)
    meta_data: pd.DataFrame  # Metadata with cell information
    peak_info: Optional[Dict] = None  # Gene-peak pairs information
    peak_info_list: Optional[List] = None  # List of gene-peak pairs
    covariates: List[str] = eqx.field(default_factory=list)  # Covariate names
    celltypes: str = ""  # Column name for cell types
    results: List[SCENTResult] = eqx.field(default_factory=list)  # Analysis results
    gene_names: Optional[List[str]] = None  # Row names of rna (genes)
    peak_names: Optional[List[str]] = None  # Row names of atac (peaks)
    cell_names: Optional[List[str]] = None  # Shared column names (cells)

    def __post_init__(self):
        """Description:
            Validate dimensions and required schema of the SCENT object.

        Args:
            None.

        Returns:
            None.
        """
        if self.rna.shape[1] != self.atac.shape[1]:
            raise ValueError(
                f"RNA matrix has {self.rna.shape[1]} cells but ATAC matrix has {self.atac.shape[1]} cells. "
                "They should have the same number of cells."
            )

        if not isinstance(self.meta_data, pd.DataFrame):
            raise ValueError("meta_data must be a pandas DataFrame.")

        if "cell" not in self.meta_data.columns:
            raise ValueError("meta_data must contain a 'cell' column for merge-by-cell alignment.")

        if self.celltypes and self.celltypes not in self.meta_data.columns:
            raise ValueError(f"Cell type column '{self.celltypes}' not found in meta_data.")

        missing_covariates = [covar for covar in self.covariates if covar not in self.meta_data.columns]
        if missing_covariates:
            raise ValueError(f"Covariates not found in meta_data: {missing_covariates}")

        if self.cell_names is not None:
            missing_cells = sorted(set(self.cell_names) - set(self.meta_data["cell"]))
            if missing_cells:
                raise ValueError(f"Some RNA/ATAC cells are missing in meta_data: {missing_cells[:5]}")

        if self.gene_names is not None and self.rna.shape[0] != len(self.gene_names):
            raise ValueError(
                f"gene_names length ({len(self.gene_names)}) != RNA matrix rows ({self.rna.shape[0]})"
            )

        if self.peak_names is not None and self.atac.shape[0] != len(self.peak_names):
            raise ValueError(
                f"peak_names length ({len(self.peak_names)}) != ATAC matrix rows ({self.atac.shape[0]})"
            )

        if self.cell_names is not None and self.rna.shape[1] != len(self.cell_names):
            raise ValueError(
                f"cell_names length ({len(self.cell_names)}) != matrix columns ({self.rna.shape[1]})"
            )

        if self.peak_info is not None and self.gene_names is not None and self.peak_names is not None:
            gene_set = set(self.gene_names)
            genes = self.peak_info.get("genes", [])
            if not all(gene in gene_set for gene in genes):
                raise ValueError("Some genes in peak_info are not in the RNA matrix")

            peak_set = set(self.peak_names)
            peaks = self.peak_info.get("peaks", [])
            if not all(peak in peak_set for peak in peaks):
                raise ValueError("Some peaks in peak_info are not in the ATAC matrix")

    @staticmethod
    def _iter_gene_peak_pairs(peak_info: Optional[Dict], peak_info_list: Optional[List]) -> List[Tuple[str, str]]:
        """Description:
            Return ordered gene-peak pairs, preferring peak_info row order to match the R path.

        Args:
            peak_info: Dictionary containing gene/peak lists and optional pair tuples.
            peak_info_list: Fallback list of {'gene', 'peak'} dictionaries.

        Returns:
            Ordered list of (gene, peak) tuples.
        """
        if peak_info is not None and "pairs" in peak_info:
            return [(str(gene), str(peak)) for gene, peak in peak_info["pairs"]]

        if peak_info_list:
            return [(str(item["gene"]), str(item["peak"])) for item in peak_info_list]

        return []

    @staticmethod
    def _encode_covariates(df2: pd.DataFrame, covariates: Sequence[str]) -> pd.DataFrame:
        """Description:
            Build predictor columns with R-like order: atac first, then covariates.

        Args:
            df2: Cell type filtered data table with atac and covariate columns.
            covariates: Covariate column names to encode.

        Returns:
            DataFrame of encoded predictor columns.
        """
        predictors = pd.DataFrame(index=df2.index)
        predictors["atac"] = pd.to_numeric(df2["atac"], errors="coerce")

        for covar in covariates:
            series = df2[covar]
            if is_bool_dtype(series) or is_numeric_dtype(series):
                predictors[covar] = pd.to_numeric(series, errors="coerce")
            else:
                # Approximate R treatment-contrast coding: sorted levels, first level as baseline.
                series_str = series.astype("string")
                levels = sorted(series_str.dropna().unique().tolist())
                for level in levels[1:]:
                    col_name = f"{covar}{level}"
                    predictors[col_name] = (series_str == level).astype(float)

        return predictors

    @classmethod
    def _build_design_matrix(cls, df2: pd.DataFrame, covariates: Sequence[str]) -> Tuple[jnp.ndarray, jnp.ndarray, List[str]]:
        """Description:
            Construct model inputs aligned with exprs ~ atac + covariates.

        Args:
            df2: Cell type filtered data table.
            covariates: Covariate column names to include in the model.

        Returns:
            Tuple of design matrix X, response vector y, and feature names.
        """
        y_series = pd.to_numeric(df2["exprs"], errors="coerce").rename("exprs")
        X_df = cls._encode_covariates(df2, covariates)
        model_df = pd.concat([y_series, X_df], axis=1).dropna()

        if model_df.empty:
            return jnp.zeros((0, 0)), jnp.zeros((0, 1)), []

        feature_df = model_df.drop(columns=["exprs"])
        feature_names = ["(Intercept)"] + feature_df.columns.tolist()

        X_np = np.column_stack(
            [
                np.ones(feature_df.shape[0], dtype=float),
                feature_df.to_numpy(dtype=float),
            ]
        )
        y_np = model_df["exprs"].to_numpy(dtype=float).reshape(-1, 1)

        X = jnp.asarray(X_np)
        y = jnp.asarray(y_np)

        return X, y, feature_names

    @staticmethod
    def _sparse_row_to_dense(mat, row_idx: int) -> np.ndarray:
        """Description:
            Extract a single row from a sparse matrix as a dense 1-D numpy array.
            Mirrors R's ``object@rna[gene,]`` followed by ``as.numeric()``.

        Args:
            mat: Sparse matrix (CSR format preferred for efficient row slicing).
            row_idx: Integer row index.

        Returns:
            1-D numpy array of float64.
        """
        row = mat[row_idx]
        if sparse.issparse(row):
            return np.asarray(row.toarray(), dtype=np.float64).ravel()
        return np.asarray(row, dtype=np.float64).ravel()

    def run_scent(
        self,
        celltype: str,
        ncores: int = 1,
        regr: str = "poisson",
        bin_atac: bool = True,
        bootstrap_samples: int = 100,
        min_nonzero_frac: float = 0.05,
        key: Optional[rdm.PRNGKey] = None,
        device: str = "auto",
    ) -> List[SCENTResult]:
        """Description:
            Run SCENT with a code path aligned to the original R implementation.
            ncores is retained for API compatibility; runtime parallelism is controlled by JAX.

        Args:
            celltype: Target cell type label used for filtering.
            ncores: Compatibility argument; currently not used for explicit threading.
            regr: Regression family, either 'poisson' or 'negbin'.
            bin_atac: Whether to binarize ATAC counts (>0 -> 1).
            bootstrap_samples: Initial bootstrap sample size (stage-1).
            min_nonzero_frac: Minimum fraction of cells with non-zero expression (RNA) and
                non-zero accessibility (ATAC) required to test a gene-peak pair. Default 0.05.
            key: Optional JAX PRNG key for reproducibility.
            device: JAX backend to use. ``"auto"`` (default) selects gpu → tpu → cpu
                in priority order. Also accepts ``"cpu"``, ``"gpu"``, or ``"tpu"``.

        Returns:
            List of SCENTResult entries for tested gene-peak pairs.
        """
        del ncores

        if key is None:
            key = rdm.PRNGKey(0)

        if regr == "poisson":
            family = Poisson()
        elif regr == "negbin":
            family = NegativeBinomial()
        else:
            raise ValueError(f"Regression type {regr} not supported. Use 'poisson' or 'negbin'.")

        if self.gene_names is None or self.peak_names is None or self.cell_names is None:
            raise ValueError("gene_names, peak_names, and cell_names are required for run_scent.")

        target_device = _resolve_device(device)

        # Build name -> row-index lookup dicts (analogous to R Dimnames indexing)
        gene_to_idx = {name: i for i, name in enumerate(self.gene_names)}
        peak_to_idx = {name: i for i, name in enumerate(self.peak_names)}

        res: List[SCENTResult] = []
        gene_peak_pairs = self._iter_gene_peak_pairs(self.peak_info, self.peak_info_list)

        # jax.default_device ensures all jnp.asarray / jax.vmap / GLM ops
        # are dispatched to target_device without modifying downstream code.
        with jax.default_device(target_device):
            for gene, this_peak in gene_peak_pairs:
                if gene not in gene_to_idx or this_peak not in peak_to_idx:
                    continue

                # Extract single peak row: sparse -> dense (R: object@atac[this_peak,])
                atac_vec = self._sparse_row_to_dense(self.atac, peak_to_idx[this_peak])
                atac_target = pd.DataFrame({"cell": self.cell_names, "atac": atac_vec})

                if bin_atac:
                    atac_target.loc[atac_target["atac"] > 0, "atac"] = 1.0

                # Extract single gene row: sparse -> dense (R: object@rna[gene,])
                exprs_vec = self._sparse_row_to_dense(self.rna, gene_to_idx[gene])
                df = pd.DataFrame({"cell": self.cell_names, "exprs": exprs_vec})
                df = df.merge(atac_target, on="cell")
                df = df.merge(self.meta_data, on="cell")

                df2 = df[df[self.celltypes] == celltype]
                if df2.empty:
                    continue

                nonzero_m = (df2["exprs"] > 0).mean()
                nonzero_a = (df2["atac"] > 0).mean()
                if not (nonzero_m > min_nonzero_frac and nonzero_a > min_nonzero_frac):
                    continue

                X, y, feature_names = self._build_design_matrix(df2, self.covariates)
                if X.shape[0] == 0 or "atac" not in feature_names:
                    continue

                glm = GLM(family=family)
                eta, alpha_n = glm.calc_eta_and_dispersion(X, y, jnp.zeros_like(y))
                glm_state = glm.fit(X, y, init=eta, alpha_init=alpha_n, se_estimator=FisherInfoError())

                atac_idx = feature_names.index("atac")
                coef = float(glm_state.beta[atac_idx].item())
                se = float(glm_state.se[atac_idx].item())
                z = float(glm_state.z[atac_idx].item())
                p = float(glm_state.p[atac_idx].item())

                key, pair_key = rdm.split(key)
                p0 = bootstrap_test(
                    X=X,
                    y=y,
                    family=family,
                    initial_samples=bootstrap_samples,
                    key=pair_key,
                    atac_idx=atac_idx,
                    obs_coef=coef,
                )

                res.append(
                    SCENTResult(
                        gene=gene,
                        peak=this_peak,
                        beta=coef,
                        se=se,
                        z=z,
                        p=p,
                        boot_basic_p=p0,
                    )
                )

        return res
