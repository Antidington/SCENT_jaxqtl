# -*- coding: utf-8 -*-
# @Time       : 2025/9/26 14:54
# @Author     : Jichen Breeze Wen
# E-mail      : Breezewjc952@gmail.com
# @Description: Utility functions for SCENT

import os
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
import pandas as pd
from jaxtyping import ArrayLike


def create_peak_to_gene_list(
    atac_peaks: List[str],
    gene_bed_path: str,
    temp_dir: str = "./temp",
    nbatch: int = 10
) -> List[Dict]:
    """Create list of gene-peak pairs for parallelization
    
    Args:
        atac_peaks: List of peak names
        gene_bed_path: Path to gene bed file with 500kb windows
        temp_dir: Directory for temporary files
        nbatch: Number of batches to create
        
    Returns:
        List of gene-peak pairs grouped into batches
    """
    import subprocess
    
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    # Write peaks to temporary bed file
    temp_peak_file = os.path.join(temp_dir, "temp_peaks.bed")
    with open(temp_peak_file, "w") as f:
        for peak in atac_peaks:
            # Parse peak name (format: chr-start-end)
            parts = peak.replace(":", "-").replace("_", "-").split("-")
            if len(parts) >= 3:
                chrom, start, end = parts[0], parts[1], parts[2]
                f.write(f"{chrom}\t{start}\t{end}\t{peak}\n")
    
    # Intersect peaks with gene windows
    temp_intersect_file = os.path.join(temp_dir, "temp_intersect.bed")
    cmd = f"bedtools intersect -a {gene_bed_path} -b {temp_peak_file} -wa -wb -loj | gzip -c > {temp_intersect_file}.gz"
    subprocess.run(cmd, shell=True, check=True)
    
    # Read intersection results
    df = pd.read_csv(f"{temp_intersect_file}.gz", sep="\t", header=None)
    
    # Filter out non-overlapping entries
    df = df[df.iloc[:, 4] != "."]
    
    # Extract gene-peak pairs
    gene_peak_pairs = []
    for _, row in df.iterrows():
        gene = row[3]  # Gene is in the 4th column
        peak = row[7]  # Peak is in the 8th column
        gene_peak_pairs.append({"gene": gene, "peak": peak})
    
    # Split into batches
    batch_size = max(1, len(gene_peak_pairs) // nbatch)
    batches = []
    for i in range(0, len(gene_peak_pairs), batch_size):
        batches.append(gene_peak_pairs[i:i+batch_size])
    
    # Clean up temporary files
    os.remove(temp_peak_file)
    os.remove(f"{temp_intersect_file}.gz")
    
    return batches


def preprocess_data(
    rna_matrix: ArrayLike,
    atac_matrix: ArrayLike,
    meta_data: pd.DataFrame,
    min_cells_per_gene: int = 10,
    min_cells_per_peak: int = 10
) -> Tuple[ArrayLike, ArrayLike, pd.DataFrame]:
    """Preprocess RNA and ATAC data
    
    Args:
        rna_matrix: RNA expression matrix (genes x cells)
        atac_matrix: ATAC accessibility matrix (peaks x cells)
        meta_data: Metadata with cell information
        min_cells_per_gene: Minimum number of cells expressing a gene
        min_cells_per_peak: Minimum number of cells with peak accessibility
        
    Returns:
        Filtered RNA matrix, ATAC matrix, and metadata
    """
    # Filter genes by minimum number of expressing cells
    gene_counts = (rna_matrix > 0).sum(axis=1)
    genes_to_keep = gene_counts >= min_cells_per_gene
    rna_filtered = rna_matrix[genes_to_keep, :]
    
    # Filter peaks by minimum number of accessible cells
    peak_counts = (atac_matrix > 0).sum(axis=1)
    peaks_to_keep = peak_counts >= min_cells_per_peak
    atac_filtered = atac_matrix[peaks_to_keep, :]
    
    return rna_filtered, atac_filtered, meta_data