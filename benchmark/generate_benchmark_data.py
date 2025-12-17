#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate benchmark datasets for SCENT comparison
This script generates datasets of various sizes to test performance and consistency
"""

import os
import numpy as np
import pandas as pd
from scipy import sparse
import argparse


def generate_benchmark_data(
    output_dir="data",
    dataset_name="small",
    n_genes=100,
    n_peaks=500,
    n_cells=1000,
    n_pairs=200,
    n_true_assoc=30,
    seed=42
):
    """Generate benchmark data for SCENT comparison

    Args:
        output_dir: Output directory
        dataset_name: Name of the dataset (small, medium, large)
        n_genes: Number of genes
        n_peaks: Number of peaks
        n_cells: Number of cells
        n_pairs: Number of gene-peak pairs to test
        n_true_assoc: Number of true associations to embed
        seed: Random seed
    """
    # Create output directory if it doesn't exist
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Set random seed for reproducibility
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"Generating {dataset_name} dataset")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  - Genes: {n_genes}")
    print(f"  - Peaks: {n_peaks}")
    print(f"  - Cells: {n_cells}")
    print(f"  - Gene-peak pairs: {n_pairs}")
    print(f"  - True associations: {n_true_assoc}")
    print(f"  - Output: {dataset_dir}")

    # Generate gene names
    gene_names = [f"GENE{i:06d}" for i in range(n_genes)]

    # Generate peak names (format: chr-start-end)
    chromosomes = [f"chr{i}" for i in range(1, 23)]
    peak_names = []
    for i in range(n_peaks):
        chrom = np.random.choice(chromosomes)
        start = np.random.randint(1000000, 250000000)
        end = start + np.random.randint(200, 2000)
        peak_names.append(f"{chrom}-{start}-{end}")

    # Generate cell IDs
    cell_ids = [f"CELL_{i:06d}" for i in range(n_cells)]

    # Generate RNA expression matrix (negative binomial distributed)
    print("  Generating RNA expression matrix...")
    # Use negative binomial to simulate count data
    # Parameters chosen to match real scRNA-seq data characteristics
    rna_matrix = np.random.negative_binomial(2, 0.7, size=(n_genes, n_cells))
    # Make it sparse (70% zeros) to match real data
    zero_mask = np.random.random((n_genes, n_cells)) < 0.70
    rna_matrix[zero_mask] = 0

    # Generate ATAC accessibility matrix (sparse binary)
    print("  Generating ATAC accessibility matrix...")
    # ATAC data is typically more sparse (90% zeros)
    atac_matrix = (np.random.random((n_peaks, n_cells)) > 0.90).astype(int)

    # Generate metadata with batch effects
    print("  Generating metadata...")
    cell_types = ["T_cell", "B_cell", "Monocyte", "NK_cell"]
    cell_type_probs = [0.45, 0.30, 0.15, 0.10]

    metadata = pd.DataFrame({
        "cell": cell_ids,
        "cell_type": np.random.choice(cell_types, size=n_cells, p=cell_type_probs),
        "batch": np.random.choice(["batch1", "batch2", "batch3"], size=n_cells),
        "n_counts": np.random.randint(1000, 10000, size=n_cells),
        "percent_mito": np.random.uniform(0.01, 0.15, size=n_cells)
    })

    # Generate initial gene-peak pairs randomly
    print("  Generating gene-peak pairs...")
    selected_genes = np.random.choice(gene_names, size=n_pairs, replace=True)
    selected_peaks = np.random.choice(peak_names, size=n_pairs, replace=True)

    peak_info = pd.DataFrame({
        "gene": selected_genes,
        "peak": selected_peaks
    })
    # Remove duplicates
    peak_info = peak_info.drop_duplicates()

    # Create true associations with strong correlations
    print(f"  Embedding {n_true_assoc} true associations...")
    true_associations = []

    for i in range(n_true_assoc):
        # Select random gene and peak
        gene_idx = np.random.randint(0, n_genes)
        peak_idx = np.random.randint(0, n_peaks)
        gene = gene_names[gene_idx]
        peak = peak_names[peak_idx]

        # Get T cell indices (main cell type for associations)
        t_cell_mask = metadata["cell_type"] == "T_cell"
        t_cell_indices = np.where(t_cell_mask)[0]

        # Create strong association: when peak is accessible, gene expression increases
        for cell_idx in t_cell_indices:
            if atac_matrix[peak_idx, cell_idx] > 0:
                # Strong increase in gene expression when peak is accessible
                # Add Poisson noise to make it realistic
                increase = np.random.poisson(50) + 20
                rna_matrix[gene_idx, cell_idx] += increase
            else:
                # Low expression when peak is not accessible
                rna_matrix[gene_idx, cell_idx] = np.random.poisson(2)

        true_associations.append({
            "gene": gene,
            "peak": peak,
            "gene_idx": gene_idx,
            "peak_idx": peak_idx
        })

        # Add this pair to peak_info if not already there
        pair_exists = ((peak_info["gene"] == gene) & (peak_info["peak"] == peak)).any()
        if not pair_exists:
            new_pair = pd.DataFrame({"gene": [gene], "peak": [peak]})
            peak_info = pd.concat([peak_info, new_pair], ignore_index=True)

    # Save data in CSV format
    print("  Saving data files...")

    # RNA matrix
    rna_df = pd.DataFrame(rna_matrix, index=gene_names, columns=cell_ids)
    rna_df.to_csv(os.path.join(dataset_dir, "rna_matrix.csv"))

    # ATAC matrix
    atac_df = pd.DataFrame(atac_matrix, index=peak_names, columns=cell_ids)
    atac_df.to_csv(os.path.join(dataset_dir, "atac_matrix.csv"))

    # Metadata
    metadata.to_csv(os.path.join(dataset_dir, "metadata.csv"), index=False)

    # Peak info
    peak_info.to_csv(os.path.join(dataset_dir, "peak_info.csv"), index=False)

    # True associations (for validation)
    true_assoc_df = pd.DataFrame(true_associations)
    true_assoc_df.to_csv(os.path.join(dataset_dir, "true_associations.csv"), index=False)

    # Also save sparse matrices for R
    print("  Saving sparse matrices for R...")
    rna_sparse = sparse.csr_matrix(rna_matrix)
    atac_sparse = sparse.csr_matrix(atac_matrix)

    sparse.save_npz(os.path.join(dataset_dir, "rna_matrix_sparse.npz"), rna_sparse)
    sparse.save_npz(os.path.join(dataset_dir, "atac_matrix_sparse.npz"), atac_sparse)

    # Save dimension info for R
    with open(os.path.join(dataset_dir, "gene_names.txt"), "w") as f:
        f.write("\n".join(gene_names))

    with open(os.path.join(dataset_dir, "peak_names.txt"), "w") as f:
        f.write("\n".join(peak_names))

    with open(os.path.join(dataset_dir, "cell_names.txt"), "w") as f:
        f.write("\n".join(cell_ids))

    # Print summary statistics
    print(f"\n  Dataset statistics:")
    print(f"    RNA matrix: {rna_df.shape[0]} genes × {rna_df.shape[1]} cells")
    print(f"    - Sparsity: {(rna_matrix == 0).sum() / rna_matrix.size * 100:.1f}%")
    print(f"    - Mean expression: {rna_matrix.mean():.2f}")
    print(f"    - Max expression: {rna_matrix.max()}")

    print(f"    ATAC matrix: {atac_df.shape[0]} peaks × {atac_df.shape[1]} cells")
    print(f"    - Sparsity: {(atac_matrix == 0).sum() / atac_matrix.size * 100:.1f}%")
    print(f"    - Accessible peaks: {atac_matrix.sum()}")

    print(f"    Metadata: {metadata.shape[0]} cells")
    for ct in cell_types:
        count = (metadata["cell_type"] == ct).sum()
        print(f"      - {ct}: {count} cells ({count/n_cells*100:.1f}%)")

    print(f"    Gene-peak pairs: {peak_info.shape[0]}")
    print(f"    True associations: {len(true_associations)}")
    print(f"\n  ✓ Dataset saved to: {dataset_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark datasets for SCENT comparison"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["small", "medium", "large"],
        choices=["small", "medium", "large", "all"],
        help="Which datasets to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Define dataset configurations
    configs = {
        "small": {
            "n_genes": 100,
            "n_peaks": 300,
            "n_cells": 500,
            "n_pairs": 150,
            "n_true_assoc": 20
        },
        "medium": {
            "n_genes": 500,
            "n_peaks": 1500,
            "n_cells": 2000,
            "n_pairs": 500,
            "n_true_assoc": 50
        },
        "large": {
            "n_genes": 2000,
            "n_peaks": 5000,
            "n_cells": 5000,
            "n_pairs": 1000,
            "n_true_assoc": 100
        }
    }

    # Determine which datasets to generate
    if "all" in args.datasets:
        datasets_to_generate = ["small", "medium", "large"]
    else:
        datasets_to_generate = args.datasets

    print("\n" + "="*60)
    print("SCENT Benchmark Data Generation")
    print("="*60)
    print(f"Generating datasets: {', '.join(datasets_to_generate)}")
    print(f"Random seed: {args.seed}")

    # Generate each dataset
    for dataset_name in datasets_to_generate:
        config = configs[dataset_name]
        generate_benchmark_data(
            output_dir=args.output_dir,
            dataset_name=dataset_name,
            seed=args.seed,
            **config
        )

    print("\n" + "="*60)
    print("✓ All datasets generated successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
