# -*- coding: utf-8 -*-
# @Time       : 2025/9/26 14:54
# @Author     : Jichen Breeze Wen
# E-mail      : Breezewjc952@gmail.com
# @Description: I/O functions for SCENT

from typing import List, Optional

import pandas as pd
from scipy import sparse

from .core import SCENTObject, SCENTResult


def read_matrix(file_path: str, sparse_format: bool = True):
    """Read matrix from file."""
    if file_path.endswith(".mtx"):
        mat = sparse.load_npz(file_path)
    elif file_path.endswith(".h5ad"):
        import anndata

        adata = anndata.read_h5ad(file_path)
        mat = adata.X
    else:
        sep = "\t" if file_path.endswith(".tsv") else ","
        df = pd.read_csv(file_path, sep=sep, index_col=0)
        if sparse_format:
            mat = sparse.csr_matrix(df.values)
        else:
            mat = df

    return mat


def create_scent_object(
    rna_matrix: str,
    atac_matrix: str,
    meta_data: str,
    peak_info: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    celltype_col: str = "cell_type",
) -> SCENTObject:
    """Create SCENTObject from files."""
    covariates = covariates or []

    sep = "\t" if rna_matrix.endswith(".tsv") else ","
    if rna_matrix.endswith(".mtx") or rna_matrix.endswith(".h5ad"):
        rna = read_matrix(rna_matrix, sparse_format=False)
    else:
        rna = pd.read_csv(rna_matrix, sep=sep, index_col=0)

    sep = "\t" if atac_matrix.endswith(".tsv") else ","
    if atac_matrix.endswith(".mtx") or atac_matrix.endswith(".h5ad"):
        atac = read_matrix(atac_matrix, sparse_format=False)
    else:
        atac = pd.read_csv(atac_matrix, sep=sep, index_col=0)

    if not isinstance(rna, pd.DataFrame) or not isinstance(atac, pd.DataFrame):
        raise ValueError("Strict R-path alignment requires tabular RNA/ATAC input with row and cell names.")

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

    if not rna.columns.equals(atac.columns):
        raise ValueError("RNA and ATAC matrices must have identical ordered cell columns.")

    missing_meta = sorted(set(rna.columns) - set(meta_df["cell"]))
    if missing_meta:
        raise ValueError(f"Cells missing in metadata: {missing_meta[:5]}")

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
        gene_names=rna.index.astype(str).tolist(),
        peak_names=atac.index.astype(str).tolist(),
    )


def write_results(results: List[SCENTResult], output_file: str) -> None:
    """Write SCENT results to file."""
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
