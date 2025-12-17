# -*- coding: utf-8 -*-
# @Time       : 2025/9/26 14:54
# @Author     : Jichen Breeze Wen
# E-mail      : Breezewjc952@gmail.com
# @Description: I/O functions for SCENT

import os
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import sparse

from .core import SCENTObject, SCENTResult


def read_matrix(file_path: str, sparse_format: bool = True) -> sparse.csr_matrix:
    """Read matrix from file
    
    Args:
        file_path: Path to matrix file
        sparse_format: Whether to return as sparse matrix
        
    Returns:
        Matrix as sparse.csr_matrix or numpy.ndarray
    """
    if file_path.endswith('.mtx'):
        # Read MTX format
        mat = sparse.load_npz(file_path)
    elif file_path.endswith('.h5ad'):
        # Read H5AD format (AnnData)
        import anndata
        adata = anndata.read_h5ad(file_path)
        mat = adata.X
    else:
        # Read CSV/TSV format
        sep = '\t' if file_path.endswith('.tsv') else ','
        df = pd.read_csv(file_path, sep=sep, index_col=0)
        if sparse_format:
            mat = sparse.csr_matrix(df.values)
        else:
            mat = df.values
    
    return mat


def create_scent_object(
        rna_matrix: str,
        atac_matrix: str,
        meta_data: str,
        peak_info: Optional[str] = None,
        covariates: List[str] = [],
        celltype_col: str = "cell_type"
) -> SCENTObject:
    """Create SCENTObject from files

    Args:
        rna_matrix: Path to RNA matrix file
        atac_matrix: Path to ATAC matrix file
        meta_data: Path to metadata file
        peak_info: Path to peak info file
        covariates: List of covariate names
        celltype_col: Column name for cell types

    Returns:
        SCENTObject
    """
    # Read RNA matrix - keep as DataFrame for easier indexing
    sep = '\t' if rna_matrix.endswith('.tsv') else ','
    if rna_matrix.endswith('.mtx') or rna_matrix.endswith('.h5ad'):
        rna = read_matrix(rna_matrix)
        gene_names = None  # Need to extract from feature data if available
    else:
        rna = pd.read_csv(rna_matrix, sep=sep, index_col=0)
        gene_names = rna.index.tolist()

    # Read ATAC matrix - keep as DataFrame for easier indexing
    sep = '\t' if atac_matrix.endswith('.tsv') else ','
    if atac_matrix.endswith('.mtx') or atac_matrix.endswith('.h5ad'):
        atac = read_matrix(atac_matrix)
        peak_names = None  # Need to extract from feature data if available
    else:
        atac = pd.read_csv(atac_matrix, sep=sep, index_col=0)
        peak_names = atac.index.tolist()

    # Read metadata
    sep = '\t' if meta_data.endswith('.tsv') else ','
    meta_df = pd.read_csv(meta_data, sep=sep)

    # Read peak info if provided
    peak_info_dict = None
    peak_info_list = None
    if peak_info is not None:
        sep = '\t' if peak_info.endswith('.tsv') else ','
        peak_df = pd.read_csv(peak_info, sep=sep)
        peak_info_dict = {
            'genes': peak_df.iloc[:, 0].tolist(),
            'peaks': peak_df.iloc[:, 1].tolist(),
            'pairs': list(zip(peak_df.iloc[:, 0], peak_df.iloc[:, 1]))
        }

        # Create list of gene-peak pairs for processing
        peak_info_list = [
            {'gene': gene, 'peak': peak}
            for gene, peak in zip(peak_df.iloc[:, 0], peak_df.iloc[:, 1])
        ]

    # Create SCENTObject - keep metadata as DataFrame for easier boolean indexing
    scent_obj = SCENTObject(
        rna=rna,
        atac=atac,
        meta_data=meta_df,  # Keep as DataFrame
        peak_info=peak_info_dict,
        peak_info_list=peak_info_list,
        covariates=covariates,
        celltypes=celltype_col,
        gene_names=gene_names,
        peak_names=peak_names
    )

    return scent_obj


def write_results(results: List[SCENTResult], output_file: str) -> None:
    """Write SCENT results to file

    Args:
        results: List of SCENT results
        output_file: Path to output file
    """
    # Convert results to DataFrame
    data = []
    for res in results:
        data.append({
            'gene': res.gene,
            'peak': res.peak,
            'beta': res.beta,
            'se': res.se,
            'z': res.z,
            'p': res.p,
            'boot_basic_p': res.boot_basic_p
        })

    df = pd.DataFrame(data)

    # Write to file
    sep = '\t' if output_file.endswith('.tsv') else ','
    df.to_csv(output_file, sep=sep, index=False)