#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark script for SCENT_jaxQTL
Runs SCENT_jaxQTL on benchmark datasets and measures performance
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import pandas as pd
import jax
import jax.random as random

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from SCENT_jaxqtl import io


def run_benchmark(
    data_dir,
    output_dir,
    dataset_name,
    cell_type="T_cell",
    regression="poisson",
    bootstrap_samples=100,
    seed=42
):
    """Run SCENT_jaxQTL benchmark on a dataset

    Args:
        data_dir: Directory containing the dataset
        output_dir: Directory for output results
        dataset_name: Name of the dataset
        cell_type: Cell type to analyze
        regression: Regression type (poisson or negbin)
        bootstrap_samples: Initial number of bootstrap samples
        seed: Random seed

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Running SCENT_jaxQTL on {dataset_name} dataset")
    print(f"{'='*60}")

    # Set up paths
    dataset_path = os.path.join(data_dir, dataset_name)
    rna_file = os.path.join(dataset_path, "rna_matrix.csv")
    atac_file = os.path.join(dataset_path, "atac_matrix.csv")
    metadata_file = os.path.join(dataset_path, "metadata.csv")
    peak_info_file = os.path.join(dataset_path, "peak_info.csv")

    # Check if files exist
    for f in [rna_file, atac_file, metadata_file, peak_info_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Required file not found: {f}")

    print(f"Data files:")
    print(f"  - RNA: {rna_file}")
    print(f"  - ATAC: {atac_file}")
    print(f"  - Metadata: {metadata_file}")
    print(f"  - Peak info: {peak_info_file}")

    # Set random seed
    key = random.PRNGKey(seed)

    # Measure time for data loading
    print(f"\n[1/3] Loading data...")
    load_start = time.time()

    try:
        scent_obj = io.create_scent_object(
            rna_matrix=rna_file,
            atac_matrix=atac_file,
            meta_data=metadata_file,
            peak_info=peak_info_file,
            covariates=["batch", "n_counts"],
            celltype_col="cell_type"
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

    load_time = time.time() - load_start
    print(f"  ✓ Data loaded in {load_time:.2f} seconds")

    # Get dataset statistics
    n_pairs = len(scent_obj.peak_info_list) if scent_obj.peak_info_list else 0
    print(f"\nDataset info:")
    print(f"  - RNA matrix shape: {scent_obj.rna.shape}")
    print(f"  - ATAC matrix shape: {scent_obj.atac.shape}")
    print(f"  - Gene-peak pairs: {n_pairs}")

    # Measure time for SCENT analysis
    print(f"\n[2/3] Running SCENT algorithm...")
    print(f"  - Cell type: {cell_type}")
    print(f"  - Regression: {regression}")
    print(f"  - Bootstrap samples: {bootstrap_samples}")

    analysis_start = time.time()

    try:
        results = scent_obj.run_scent(
            celltype=cell_type,
            regr=regression,
            bootstrap_samples=bootstrap_samples,
            key=key
        )
    except Exception as e:
        print(f"Error running SCENT: {e}")
        raise

    analysis_time = time.time() - analysis_start
    print(f"  ✓ Analysis completed in {analysis_time:.2f} seconds")

    # Save results
    print(f"\n[3/3] Saving results...")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        f"scent_jaxqtl_{dataset_name}_{cell_type}.csv"
    )

    save_start = time.time()
    io.write_results(results, output_file)
    save_time = time.time() - save_start

    print(f"  ✓ Results saved to: {output_file}")
    print(f"  ✓ Save time: {save_time:.2f} seconds")

    # Calculate statistics
    total_time = load_time + analysis_time + save_time

    if results:
        results_df = pd.DataFrame([{
            'gene': r.gene,
            'peak': r.peak,
            'beta': r.beta,
            'se': r.se,
            'z': r.z,
            'p': r.p,
            'boot_basic_p': r.boot_basic_p
        } for r in results])

        # Count significant results at different thresholds
        sig_001 = (results_df['boot_basic_p'] < 0.01).sum()
        sig_005 = (results_df['boot_basic_p'] < 0.05).sum()
        sig_010 = (results_df['boot_basic_p'] < 0.10).sum()

        print(f"\nResults summary:")
        print(f"  - Total associations tested: {len(results)}")
        print(f"  - Significant at p < 0.01: {sig_001}")
        print(f"  - Significant at p < 0.05: {sig_005}")
        print(f"  - Significant at p < 0.10: {sig_010}")

        # Show top 5 results
        print(f"\n  Top 5 associations:")
        top_results = results_df.nsmallest(5, 'boot_basic_p')
        for i, row in top_results.iterrows():
            print(f"    {i+1}. {row['gene']} - {row['peak']}: "
                  f"β={row['beta']:.3f}, p={row['boot_basic_p']:.6f}")

    else:
        sig_001 = sig_005 = sig_010 = 0
        print(f"\n  No significant results found")

    # Prepare benchmark statistics
    benchmark_stats = {
        "method": "SCENT_jaxQTL",
        "dataset": dataset_name,
        "cell_type": cell_type,
        "regression": regression,
        "bootstrap_samples": bootstrap_samples,
        "n_genes": scent_obj.rna.shape[0],
        "n_peaks": scent_obj.atac.shape[0],
        "n_cells": scent_obj.rna.shape[1],
        "n_pairs_tested": n_pairs,
        "n_results": len(results),
        "n_sig_001": int(sig_001),
        "n_sig_005": int(sig_005),
        "n_sig_010": int(sig_010),
        "load_time": load_time,
        "analysis_time": analysis_time,
        "save_time": save_time,
        "total_time": total_time,
        "seed": seed,
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend()
    }

    print(f"\n{'='*60}")
    print(f"Benchmark Summary")
    print(f"{'='*60}")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"    - Data loading: {load_time:.2f}s ({load_time/total_time*100:.1f}%)")
    print(f"    - Analysis: {analysis_time:.2f}s ({analysis_time/total_time*100:.1f}%)")
    print(f"    - Saving: {save_time:.2f}s ({save_time/total_time*100:.1f}%)")
    print(f"  JAX backend: {jax.default_backend()}")
    print(f"{'='*60}\n")

    return benchmark_stats


def main():
    parser = argparse.ArgumentParser(
        description="Run SCENT_jaxQTL benchmark"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing benchmark datasets"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/scent_jaxqtl",
        help="Output directory for results"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["small"],
        help="Which datasets to run (small, medium, large)"
    )
    parser.add_argument(
        "--cell_type",
        type=str,
        default="T_cell",
        help="Cell type to analyze"
    )
    parser.add_argument(
        "--regression",
        type=str,
        default="poisson",
        choices=["poisson", "negbin"],
        help="Regression type"
    )
    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        default=100,
        help="Initial number of bootstrap samples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("SCENT_jaxQTL Benchmark")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Datasets: {', '.join(args.datasets)}")
    print(f"  - Cell type: {args.cell_type}")
    print(f"  - Regression: {args.regression}")
    print(f"  - Bootstrap samples: {args.bootstrap_samples}")
    print(f"  - Random seed: {args.seed}")
    print(f"  - JAX backend: {jax.default_backend()}")

    # Run benchmarks
    all_stats = []

    for dataset in args.datasets:
        try:
            stats = run_benchmark(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                dataset_name=dataset,
                cell_type=args.cell_type,
                regression=args.regression,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed
            )
            all_stats.append(stats)

        except Exception as e:
            print(f"\nError processing {dataset}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save benchmark statistics
    if all_stats:
        stats_file = os.path.join(args.output_dir, "benchmark_stats.json")
        with open(stats_file, "w") as f:
            json.dump(all_stats, f, indent=2)
        print(f"\n✓ Benchmark statistics saved to: {stats_file}")

        # Also save as CSV for easy viewing
        stats_df = pd.DataFrame(all_stats)
        csv_file = os.path.join(args.output_dir, "benchmark_stats.csv")
        stats_df.to_csv(csv_file, index=False)
        print(f"✓ Benchmark statistics saved to: {csv_file}")

    print("\n" + "="*60)
    print("✓ SCENT_jaxQTL benchmark completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
