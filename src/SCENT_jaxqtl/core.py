# -*- coding: utf-8 -*-
# @Time       : 2025/9/26
# @Author     : Jichen Wen
# E-mail      : wenjichen.big@gmail.com / wenjichen@cncb.ac.cn
# @Description: Core SCENT functionality using jaxqtl

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as rdm
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from jaxtyping import ArrayLike

from jaxqtl.families.distribution import NegativeBinomial, Poisson
from jaxqtl.infer.glm import GLM
from jaxqtl.infer.stderr import FisherInfoError

from .bootstrap import bootstrap_test


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
    """SCENT object for single-cell epigenome and transcriptome analysis"""

    rna: ArrayLike  # RNA expression matrix (genes x cells)
    atac: ArrayLike  # ATAC accessibility matrix (peaks x cells)
    meta_data: pd.DataFrame  # Metadata with cell information
    peak_info: Optional[Dict] = None  # Gene-peak pairs information
    peak_info_list: Optional[List] = None  # List of gene-peak pairs for parallelization
    covariates: List[str] = eqx.field(default_factory=list)  # Covariate names
    celltypes: str = ""  # Column name for cell types
    results: List[SCENTResult] = eqx.field(default_factory=list)  # Analysis results
    gene_names: Optional[List[str]] = None  # List of gene names
    peak_names: Optional[List[str]] = None  # List of peak names

    def __post_init__(self):
        """Validate dimensions and schema of the input data."""
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

        if isinstance(self.rna, pd.DataFrame) and isinstance(self.atac, pd.DataFrame):
            if not self.rna.columns.equals(self.atac.columns):
                raise ValueError("RNA and ATAC matrices must share the same ordered cell columns.")

            missing_cells = sorted(set(self.rna.columns) - set(self.meta_data["cell"]))
            if missing_cells:
                raise ValueError(f"Some RNA/ATAC cells are missing in meta_data: {missing_cells[:5]}")

        if self.peak_info is not None and self.gene_names is not None and self.peak_names is not None:
            genes = self.peak_info.get("genes", [])
            if not all(gene in self.gene_names for gene in genes):
                raise ValueError("Some genes in peak_info are not in the RNA matrix")

            peaks = self.peak_info.get("peaks", [])
            if not all(peak in self.peak_names for peak in peaks):
                raise ValueError("Some peaks in peak_info are not in the ATAC matrix")

    @staticmethod
    def _iter_gene_peak_pairs(peak_info: Optional[Dict], peak_info_list: Optional[List]) -> List[Tuple[str, str]]:
        """Return ordered gene-peak pairs; prefer peak_info row order like R implementation."""
        if peak_info is not None and "pairs" in peak_info:
            return [(str(gene), str(peak)) for gene, peak in peak_info["pairs"]]

        if peak_info_list:
            return [(str(item["gene"]), str(item["peak"])) for item in peak_info_list]

        return []

    @staticmethod
    def _encode_covariates(df2: pd.DataFrame, covariates: Sequence[str]) -> pd.DataFrame:
        """Build predictor matrix with R-like order: atac first, then covariates."""
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
    def _build_design_matrix(cls, df2: pd.DataFrame, covariates: Sequence[str]) -> Tuple[ArrayLike, ArrayLike, List[str]]:
        """Construct model inputs aligned with exprs ~ atac + covariates."""
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

    def run_scent(
        self,
        celltype: str,
        ncores: int = 1,
        regr: str = "poisson",
        bin_atac: bool = True,
        bootstrap_samples: int = 100,
        key: Optional[rdm.PRNGKey] = None,
    ) -> List[SCENTResult]:
        """Run SCENT algorithm with code path aligned to the original R implementation.

        Notes:
            ncores is retained for API compatibility. Parallelism is controlled by JAX runtime.
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

        if not isinstance(self.rna, pd.DataFrame) or not isinstance(self.atac, pd.DataFrame):
            raise ValueError("run_scent currently requires RNA and ATAC as pandas DataFrames.")

        res: List[SCENTResult] = []
        gene_peak_pairs = self._iter_gene_peak_pairs(self.peak_info, self.peak_info_list)

        for gene, this_peak in gene_peak_pairs:
            if gene not in self.rna.index or this_peak not in self.atac.index:
                continue

            atac_target = pd.DataFrame(
                {
                    "cell": self.atac.columns,
                    "atac": pd.to_numeric(self.atac.loc[this_peak], errors="coerce").to_numpy(dtype=float),
                }
            )

            if bin_atac:
                atac_target.loc[atac_target["atac"] > 0, "atac"] = 1.0

            mrna_target = pd.to_numeric(self.rna.loc[gene], errors="coerce")
            df = pd.DataFrame({"cell": mrna_target.index, "exprs": mrna_target.to_numpy(dtype=float)})
            df = df.merge(atac_target, on="cell")
            df = df.merge(self.meta_data, on="cell")

            df2 = df[df[self.celltypes] == celltype]
            if df2.empty:
                continue

            nonzero_m = (df2["exprs"] > 0).mean()
            nonzero_a = (df2["atac"] > 0).mean()
            if not (nonzero_m > 0.05 and nonzero_a > 0.05):
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
