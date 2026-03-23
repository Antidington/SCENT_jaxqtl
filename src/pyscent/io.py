# -*- coding: utf-8 -*-
# @Time       : 2025/9/26 14:54
# @Author     : Jichen Breeze Wen
# E-mail      : Breezewjc952@gmail.com
# @Description: I/O functions for SCENT

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from .core import SCENTObject, SCENTResult


def read_matrix(
    file_path: str,
) -> Tuple[sparse.csr_matrix, Optional[List[str]], Optional[List[str]]]:
    """Description:
        Read a matrix file and return a sparse CSR matrix with row/column names.

        Supported formats:
          - CSV (.csv): comma-separated, first column as row names, header as column names
          - TSV (.tsv): tab-separated, same layout as CSV
          - MTX (.mtx): Matrix Market format via scipy.io.mmread (no names)
          - H5AD (.h5ad): AnnData format (cells x genes); transposed to genes x cells

    Args:
        file_path: Path to the matrix file.

    Returns:
        Tuple of (csr_matrix, row_names, col_names).
        row_names / col_names may be None when names are unavailable (e.g. MTX).
    """
    if file_path.endswith(".mtx") or file_path.endswith(".mtx.gz"):
        from scipy.io import mmread

        mat = mmread(file_path)
        return sparse.csr_matrix(mat), None, None

    if file_path.endswith(".h5ad"):
        import anndata

        adata = anndata.read_h5ad(file_path)
        # AnnData is cells(obs) x genes(var); transpose to genes x cells
        X = adata.X
        if sparse.issparse(X):
            mat = sparse.csr_matrix(X.T)
        else:
            mat = sparse.csr_matrix(np.asarray(X).T)
        row_names = adata.var_names.astype(str).tolist()
        col_names = adata.obs_names.astype(str).tolist()
        return mat, row_names, col_names

    # CSV / TSV
    sep = "\t" if file_path.endswith(".tsv") else ","
    df = pd.read_csv(file_path, sep=sep, index_col=0)
    row_names = df.index.astype(str).tolist()
    col_names = df.columns.astype(str).tolist()
    mat = sparse.csr_matrix(df.values, dtype=np.float64)
    return mat, row_names, col_names


def create_scent_object(
    rna_matrix: str,
    atac_matrix: str,
    meta_data: str,
    peak_info: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    celltype_col: str = "cell_type",
) -> SCENTObject:
    """Description:
        Create a SCENTObject from RNA, ATAC, metadata, and optional peak-info files.

        All matrix formats supported by ``read_matrix`` are accepted.  Matrices
        are stored internally as sparse CSR (analogous to R dgCMatrix) with
        separate name lists for genes, peaks, and cells.

    Args:
        rna_matrix: Path to RNA matrix file (genes x cells).
        atac_matrix: Path to ATAC matrix file (peaks x cells).
        meta_data: Path to metadata file (CSV or TSV).
        peak_info: Optional path to gene-peak pair file.
        covariates: Optional list of covariate column names.
        celltype_col: Metadata column name for cell type labels.

    Returns:
        Constructed and validated SCENTObject instance.
    """
    covariates = covariates or []

    # --- Read matrices as sparse ---
    rna, gene_names, rna_cell_names = read_matrix(rna_matrix)
    atac, peak_names, atac_cell_names = read_matrix(atac_matrix)

    # Determine cell names
    if rna_cell_names is not None and atac_cell_names is not None:
        if rna_cell_names != atac_cell_names:
            raise ValueError("RNA and ATAC matrices must have identical ordered cell columns.")
        cell_names = rna_cell_names
    elif rna_cell_names is not None:
        cell_names = rna_cell_names
    elif atac_cell_names is not None:
        cell_names = atac_cell_names
    else:
        raise ValueError(
            "Cell names could not be determined from the input matrices. "
            "Use CSV/TSV (with headers) or H5AD format to provide cell names."
        )

    if gene_names is None:
        raise ValueError(
            "Gene names could not be determined from the RNA matrix. "
            "Use CSV/TSV (with row index) or H5AD format."
        )
    if peak_names is None:
        raise ValueError(
            "Peak names could not be determined from the ATAC matrix. "
            "Use CSV/TSV (with row index) or H5AD format."
        )

    # --- Read metadata ---
    sep = "\t" if meta_data.endswith(".tsv") else ","
    meta_df = pd.read_csv(meta_data, sep=sep)

    if "cell" not in meta_df.columns:
        if "cell_id" in meta_df.columns:
            meta_df = meta_df.rename(columns={"cell_id": "cell"})
        else:
            raise ValueError("Metadata must contain a 'cell' column.")

    if celltype_col not in meta_df.columns:
        raise ValueError(f"Cell type column '{celltype_col}' not found in metadata.")

    missing_cov = [cov for cov in covariates if cov not in meta_df.columns]
    if missing_cov:
        raise ValueError(f"Covariates not found in metadata: {missing_cov}")

    missing_meta = sorted(set(cell_names) - set(meta_df["cell"]))
    if missing_meta:
        raise ValueError(f"Cells missing in metadata: {missing_meta[:5]}")

    # --- Read peak info ---
    peak_info_dict = None
    peak_info_list = None
    if peak_info is not None:
        sep = "\t" if peak_info.endswith(".tsv") else ","
        peak_df = pd.read_csv(peak_info, sep=sep)
        if peak_df.shape[1] < 2:
            raise ValueError("peak_info must have at least two columns: gene and peak")

        peak_df = peak_df.iloc[:, :2].copy()
        peak_df.columns = ["gene", "peak"]

        peak_info_dict = {
            "genes": peak_df["gene"].astype(str).tolist(),
            "peaks": peak_df["peak"].astype(str).tolist(),
            "pairs": list(zip(peak_df["gene"].astype(str), peak_df["peak"].astype(str))),
        }

        peak_info_list = [
            {"gene": gene, "peak": peak}
            for gene, peak in zip(peak_df["gene"].astype(str), peak_df["peak"].astype(str))
        ]

    return SCENTObject(
        rna=rna,
        atac=atac,
        meta_data=meta_df,
        peak_info=peak_info_dict,
        peak_info_list=peak_info_list,
        covariates=covariates,
        celltypes=celltype_col,
        gene_names=gene_names,
        peak_names=peak_names,
        cell_names=cell_names,
    )


def write_results(results: List[SCENTResult], output_file: str) -> None:
    """Description:
        Write SCENT analysis results to a delimited output file.

    Args:
        results: List of SCENTResult entries.
        output_file: Output path ending with .csv or .tsv.

    Returns:
        None.
    """
    columns = ["gene", "peak", "beta", "se", "z", "p", "boot_basic_p"]
    data = [
        {
            "gene": res.gene,
            "peak": res.peak,
            "beta": res.beta,
            "se": res.se,
            "z": res.z,
            "p": res.p,
            "boot_basic_p": res.boot_basic_p,
        }
        for res in results
    ]

    df = pd.DataFrame(data, columns=columns)
    sep = "\t" if output_file.endswith(".tsv") else ","
    df.to_csv(output_file, sep=sep, index=False)
