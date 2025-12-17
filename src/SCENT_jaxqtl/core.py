# -*- coding: utf-8 -*-
# @Time       : 2025/9/26
# @Author     : Jichen Wen
# E-mail      : wenjichen.big@gmail.com / wenjichen@cncb.ac.cn
# @Description: Core SCENT functionality using jaxqtl

from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as rdm
import pandas as pd
from jaxtyping import Array, ArrayLike

from jaxqtl.families.distribution import ExponentialFamily, NegativeBinomial, Poisson
from jaxqtl.infer.glm import GLM
from jaxqtl.infer.stderr import ErrVarEstimation, FisherInfoError

from .bootstrap import basic_p, bootstrap_test


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
    meta_data: Dict  # Metadata with cell information
    peak_info: Optional[Dict] = None  # Gene-peak pairs information
    peak_info_list: Optional[List] = None  # List of gene-peak pairs for parallelization
    covariates: List[str] = eqx.field(default_factory=list)  # Covariate names
    celltypes: str = ""  # Column name for cell types
    results: List[SCENTResult] = eqx.field(default_factory=list)  # Analysis results
    gene_names: Optional[List[str]] = None  # List of gene names
    peak_names: Optional[List[str]] = None  # List of peak names

    def __post_init__(self):
        """Validate dimensions of the input data"""
        # Check if RNA and ATAC have the same number of cells
        if self.rna.shape[1] != self.atac.shape[1]:
            raise ValueError(
                f"RNA matrix has {self.rna.shape[1]} cells but ATAC matrix has {self.atac.shape[1]} cells. "
                f"They should have the same number of cells."
            )

        # Check if peak_info genes are in gene_names
        if self.peak_info is not None and self.gene_names is not None and self.peak_names is not None:
            genes = self.peak_info.get('genes', [])
            if not all(gene in self.gene_names for gene in genes):
                raise ValueError("Some genes in peak_info are not in the RNA matrix")

            # Check if peak_info peaks are in peak_names
            peaks = self.peak_info.get('peaks', [])
            if not all(peak in self.peak_names for peak in peaks):
                raise ValueError("Some peaks in peak_info are not in the ATAC matrix")

    def run_scent(
        self,
        celltype: str,
        ncores: int = 1,
        regr: str = "poisson",
        bin_atac: bool = True,
        bootstrap_samples: int = 100,
        key: Optional[rdm.PRNGKey] = None,
    ) -> List[SCENTResult]:
        """Run SCENT algorithm on the data

        Args:
            celltype: Cell type to analyze
            ncores: Number of cores for parallelization
            regr: Regression type ("poisson" or "negbin")
            bin_atac: Whether to binarize ATAC data
            bootstrap_samples: Initial number of bootstrap samples
            key: Random key for reproducibility

        Returns:
            List of SCENT results
        """
        if key is None:
            key = rdm.PRNGKey(0)

        # Select family based on regression type
        if regr == "poisson":
            family = Poisson()
        elif regr == "negbin":
            family = NegativeBinomial()
        else:
            raise ValueError(f"Regression type {regr} not supported. Use 'poisson' or 'negbin'.")

        results = []

        # Process each gene-peak pair
        for gene_peak_pair in self.peak_info_list:
            gene = gene_peak_pair['gene']
            peak = gene_peak_pair['peak']

            # Extract data for this gene and peak
            # Both rna and atac are now DataFrames with .loc support
            gene_expr = self.rna.loc[gene].values
            peak_access = self.atac.loc[peak].values

            # Binarize ATAC data if requested
            if bin_atac:
                peak_access = (peak_access > 0).astype(float)

            # Filter by cell type
            cell_type_mask = self.meta_data[self.celltypes] == celltype

            # Get cell type specific data
            gene_expr_ct = gene_expr[cell_type_mask]
            peak_access_ct = peak_access[cell_type_mask]

            # Check if there's enough non-zero data
            nonzero_expr = (gene_expr_ct > 0).mean()
            nonzero_atac = (peak_access_ct > 0).mean()

            if nonzero_expr > 0.05 and nonzero_atac > 0.05:
                # Prepare design matrix and response
                n_samples = len(gene_expr_ct)

                X = jnp.ones((n_samples, 1))  # Intercept

                # Add covariates
                for covar in self.covariates:
                    covar_data = self.meta_data.loc[cell_type_mask, covar].values
                    # Convert categorical variables to numeric codes
                    if covar_data.dtype == object or covar_data.dtype.name == 'category':
                        covar_data = pd.Categorical(covar_data).codes.astype(float)
                    X = jnp.column_stack((X, covar_data))

                # Add ATAC as the last column
                X = jnp.column_stack((X, peak_access_ct))

                y = jnp.array(gene_expr_ct).reshape(-1, 1)

                # Run regression
                glm = GLM(family=family)

                # Fit model
                eta, alpha_n = glm.calc_eta_and_dispersion(X, y, jnp.zeros_like(y))
                glm_state = glm.fit(X, y, init=eta, alpha_init=alpha_n, se_estimator=FisherInfoError())

                # Get coefficient for ATAC
                atac_idx = X.shape[1] - 1  # Last column is ATAC
                coef = glm_state.beta[atac_idx].item()
                se = glm_state.se[atac_idx].item()
                z = glm_state.z[atac_idx].item()
                p = glm_state.p[atac_idx].item()

                # Run bootstrap if p-value is promising
                boot_p = p
                if p < 0.1:
                    # Perform bootstrap with increasing sample sizes
                    boot_p = bootstrap_test(
                        X, y, family, bootstrap_samples, key, 
                        p_threshold=p, atac_idx=atac_idx
                    )

                # Store result
                result = SCENTResult(
                    gene=gene,
                    peak=peak,
                    beta=coef,
                    se=se,
                    z=z,
                    p=p,
                    boot_basic_p=boot_p
                )
                results.append(result)

        # Don't try to modify the immutable object
        return results
