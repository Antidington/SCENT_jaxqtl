#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare benchmark results between SCENT_jaxQTL and original SCENT (R)
Analyzes runtime performance and result consistency
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_results(results_dir, method, dataset, cell_type):
    """Load results from a benchmark run

    Args:
        results_dir: Results directory
        method: Method name (scent_jaxqtl or scent_r)
        dataset: Dataset name
        cell_type: Cell type

    Returns:
        DataFrame with results
    """
    filename = f"{method}_{dataset}_{cell_type}.csv"
    filepath = os.path.join(results_dir, method, filename)

    if not os.path.exists(filepath):
        print(f"Warning: Results file not found: {filepath}")
        return None

    df = pd.read_csv(filepath)
    return df


def compare_results(jaxqtl_results, r_results, dataset_name):
    """Compare results between SCENT_jaxQTL and SCENT R

    Args:
        jaxqtl_results: DataFrame with SCENT_jaxQTL results
        r_results: DataFrame with SCENT R results
        dataset_name: Name of the dataset

    Returns:
        Dictionary with comparison statistics
    """
    print(f"\n{'='*60}")
    print(f"Comparing results for {dataset_name}")
    print(f"{'='*60}")

    if jaxqtl_results is None or r_results is None:
        print("  Error: Missing results for comparison")
        return None

    # Merge results on gene-peak pairs
    merged = pd.merge(
        jaxqtl_results,
        r_results,
        on=['gene', 'peak'],
        suffixes=('_jaxqtl', '_r'),
        how='outer',
        indicator=True
    )

    n_jaxqtl_only = (merged['_merge'] == 'left_only').sum()
    n_r_only = (merged['_merge'] == 'right_only').sum()
    n_both = (merged['_merge'] == 'both').sum()

    print(f"\nResult overlap:")
    print(f"  - SCENT_jaxQTL only: {n_jaxqtl_only}")
    print(f"  - SCENT R only: {n_r_only}")
    print(f"  - Both methods: {n_both}")

    if n_both == 0:
        print("  Warning: No overlapping results found")
        return None

    # Get overlapping results
    both_results = merged[merged['_merge'] == 'both'].copy()

    # Compare coefficients (beta)
    beta_corr = both_results['beta_jaxqtl'].corr(both_results['beta_r'])
    beta_mae = np.mean(np.abs(both_results['beta_jaxqtl'] - both_results['beta_r']))
    beta_rmse = np.sqrt(np.mean((both_results['beta_jaxqtl'] - both_results['beta_r'])**2))

    print(f"\nCoefficient (beta) comparison:")
    print(f"  - Correlation: {beta_corr:.6f}")
    print(f"  - Mean Absolute Error: {beta_mae:.6f}")
    print(f"  - Root Mean Squared Error: {beta_rmse:.6f}")

    # Compare p-values
    p_corr = stats.spearmanr(
        both_results['boot_basic_p_jaxqtl'],
        both_results['boot_basic_p_r']
    )[0]

    print(f"\nP-value comparison:")
    print(f"  - Spearman correlation: {p_corr:.6f}")

    # Compare significant results at different thresholds
    thresholds = [0.01, 0.05, 0.10]
    for thresh in thresholds:
        jaxqtl_sig = (both_results['boot_basic_p_jaxqtl'] < thresh).sum()
        r_sig = (both_results['boot_basic_p_r'] < thresh).sum()
        both_sig = ((both_results['boot_basic_p_jaxqtl'] < thresh) &
                    (both_results['boot_basic_p_r'] < thresh)).sum()

        overlap = both_sig / max(jaxqtl_sig, r_sig) * 100 if max(jaxqtl_sig, r_sig) > 0 else 0

        print(f"\n  Significant at p < {thresh}:")
        print(f"    - SCENT_jaxQTL: {jaxqtl_sig}")
        print(f"    - SCENT R: {r_sig}")
        print(f"    - Agreement: {both_sig} ({overlap:.1f}%)")

    # Create comparison statistics dictionary
    comparison_stats = {
        'dataset': dataset_name,
        'n_pairs_jaxqtl': len(jaxqtl_results),
        'n_pairs_r': len(r_results),
        'n_pairs_both': n_both,
        'n_pairs_jaxqtl_only': n_jaxqtl_only,
        'n_pairs_r_only': n_r_only,
        'beta_correlation': beta_corr,
        'beta_mae': beta_mae,
        'beta_rmse': beta_rmse,
        'pvalue_spearman': p_corr
    }

    # Add significance overlap for each threshold
    for thresh in thresholds:
        jaxqtl_sig = (both_results['boot_basic_p_jaxqtl'] < thresh).sum()
        r_sig = (both_results['boot_basic_p_r'] < thresh).sum()
        both_sig = ((both_results['boot_basic_p_jaxqtl'] < thresh) &
                    (both_results['boot_basic_p_r'] < thresh)).sum()

        comparison_stats[f'n_sig_{thresh}_jaxqtl'] = jaxqtl_sig
        comparison_stats[f'n_sig_{thresh}_r'] = r_sig
        comparison_stats[f'n_sig_{thresh}_both'] = both_sig

    return comparison_stats, both_results


def plot_comparisons(both_results, dataset_name, output_dir):
    """Create comparison plots

    Args:
        both_results: DataFrame with merged results
        dataset_name: Name of the dataset
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'SCENT Comparison: {dataset_name}', fontsize=16, fontweight='bold')

    # 1. Beta comparison scatter plot
    ax1 = axes[0, 0]
    ax1.scatter(both_results['beta_r'], both_results['beta_jaxqtl'],
                alpha=0.5, s=20)
    ax1.plot([both_results['beta_r'].min(), both_results['beta_r'].max()],
             [both_results['beta_r'].min(), both_results['beta_r'].max()],
             'r--', lw=2, label='Perfect agreement')
    ax1.set_xlabel('SCENT R: Beta coefficient', fontsize=12)
    ax1.set_ylabel('SCENT_jaxQTL: Beta coefficient', fontsize=12)
    ax1.set_title('Coefficient Comparison', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add correlation text
    corr = both_results['beta_jaxqtl'].corr(both_results['beta_r'])
    ax1.text(0.05, 0.95, f'Pearson r = {corr:.4f}',
             transform=ax1.transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. P-value comparison scatter plot (log scale)
    ax2 = axes[0, 1]
    # Add small constant to avoid log(0)
    p_jaxqtl = both_results['boot_basic_p_jaxqtl'].replace(0, 1e-10)
    p_r = both_results['boot_basic_p_r'].replace(0, 1e-10)

    ax2.scatter(np.log10(p_r), np.log10(p_jaxqtl), alpha=0.5, s=20)
    ax2.plot([np.log10(p_r).min(), np.log10(p_r).max()],
             [np.log10(p_r).min(), np.log10(p_r).max()],
             'r--', lw=2, label='Perfect agreement')
    ax2.set_xlabel('SCENT R: log10(p-value)', fontsize=12)
    ax2.set_ylabel('SCENT_jaxQTL: log10(p-value)', fontsize=12)
    ax2.set_title('P-value Comparison', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add Spearman correlation
    spearman = stats.spearmanr(
        both_results['boot_basic_p_jaxqtl'],
        both_results['boot_basic_p_r']
    )[0]
    ax2.text(0.05, 0.95, f'Spearman ρ = {spearman:.4f}',
             transform=ax2.transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 3. Beta difference vs mean
    ax3 = axes[1, 0]
    beta_mean = (both_results['beta_r'] + both_results['beta_jaxqtl']) / 2
    beta_diff = both_results['beta_jaxqtl'] - both_results['beta_r']

    ax3.scatter(beta_mean, beta_diff, alpha=0.5, s=20)
    ax3.axhline(y=0, color='r', linestyle='--', lw=2)
    ax3.axhline(y=beta_diff.mean(), color='b', linestyle='-', lw=1,
                label=f'Mean diff = {beta_diff.mean():.4f}')
    ax3.set_xlabel('Mean Beta coefficient', fontsize=12)
    ax3.set_ylabel('Beta difference (jaxQTL - R)', fontsize=12)
    ax3.set_title('Bland-Altman Plot (Beta)', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Significance agreement
    ax4 = axes[1, 1]
    thresholds = [0.01, 0.05, 0.10]
    jaxqtl_counts = []
    r_counts = []
    both_counts = []

    for thresh in thresholds:
        jaxqtl_sig = (both_results['boot_basic_p_jaxqtl'] < thresh).sum()
        r_sig = (both_results['boot_basic_p_r'] < thresh).sum()
        both_sig = ((both_results['boot_basic_p_jaxqtl'] < thresh) &
                    (both_results['boot_basic_p_r'] < thresh)).sum()

        jaxqtl_counts.append(jaxqtl_sig)
        r_counts.append(r_sig)
        both_counts.append(both_sig)

    x = np.arange(len(thresholds))
    width = 0.25

    ax4.bar(x - width, jaxqtl_counts, width, label='SCENT_jaxQTL', alpha=0.8)
    ax4.bar(x, r_counts, width, label='SCENT R', alpha=0.8)
    ax4.bar(x + width, both_counts, width, label='Both agree', alpha=0.8)

    ax4.set_xlabel('P-value threshold', fontsize=12)
    ax4.set_ylabel('Number of significant pairs', fontsize=12)
    ax4.set_title('Significance Agreement', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'p < {t}' for t in thresholds])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save plot
    plot_file = os.path.join(output_dir, f'comparison_{dataset_name}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plots saved to: {plot_file}")

    plt.close()


def plot_performance(jaxqtl_stats, r_stats, output_dir):
    """Create performance comparison plots

    Args:
        jaxqtl_stats: List of SCENT_jaxQTL benchmark stats
        r_stats: List of SCENT R benchmark stats
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert to DataFrames
    jaxqtl_df = pd.DataFrame(jaxqtl_stats)
    r_df = pd.DataFrame(r_stats)

    # Merge on dataset
    merged_df = pd.merge(
        jaxqtl_df[['dataset', 'total_time', 'analysis_time', 'n_pairs_tested']],
        r_df[['dataset', 'total_time', 'analysis_time', 'n_pairs_tested']],
        on='dataset',
        suffixes=('_jaxqtl', '_r')
    )

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Performance Comparison', fontsize=16, fontweight='bold')

    # 1. Total time comparison
    ax1 = axes[0]
    x = np.arange(len(merged_df))
    width = 0.35

    ax1.bar(x - width/2, merged_df['total_time_jaxqtl'], width,
            label='SCENT_jaxQTL', alpha=0.8)
    ax1.bar(x + width/2, merged_df['total_time_r'], width,
            label='SCENT R', alpha=0.8)

    ax1.set_xlabel('Dataset', fontsize=12)
    ax1.set_ylabel('Total time (seconds)', fontsize=12)
    ax1.set_title('Total Runtime', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(merged_df['dataset'])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add speedup annotations
    for i, row in merged_df.iterrows():
        speedup = row['total_time_r'] / row['total_time_jaxqtl']
        ax1.text(i, max(row['total_time_r'], row['total_time_jaxqtl']) * 1.05,
                f'{speedup:.1f}×', ha='center', fontsize=10, fontweight='bold')

    # 2. Analysis time comparison
    ax2 = axes[1]
    ax2.bar(x - width/2, merged_df['analysis_time_jaxqtl'], width,
            label='SCENT_jaxQTL', alpha=0.8)
    ax2.bar(x + width/2, merged_df['analysis_time_r'], width,
            label='SCENT R', alpha=0.8)

    ax2.set_xlabel('Dataset', fontsize=12)
    ax2.set_ylabel('Analysis time (seconds)', fontsize=12)
    ax2.set_title('Analysis Runtime', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(merged_df['dataset'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add speedup annotations
    for i, row in merged_df.iterrows():
        speedup = row['analysis_time_r'] / row['analysis_time_jaxqtl']
        ax2.text(i, max(row['analysis_time_r'], row['analysis_time_jaxqtl']) * 1.05,
                f'{speedup:.1f}×', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save plot
    plot_file = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Performance plots saved to: {plot_file}")

    plt.close()

    # Print speedup summary
    print(f"\n{'='*60}")
    print("Speedup Summary")
    print(f"{'='*60}")
    for i, row in merged_df.iterrows():
        dataset = row['dataset']
        total_speedup = row['total_time_r'] / row['total_time_jaxqtl']
        analysis_speedup = row['analysis_time_r'] / row['analysis_time_jaxqtl']
        print(f"\n{dataset}:")
        print(f"  - Total speedup: {total_speedup:.2f}×")
        print(f"  - Analysis speedup: {analysis_speedup:.2f}×")


def main():
    parser = argparse.ArgumentParser(
        description="Compare SCENT_jaxQTL and SCENT R benchmark results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing benchmark results"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["small"],
        help="Datasets to compare"
    )
    parser.add_argument(
        "--cell_type",
        type=str,
        default="T_cell",
        help="Cell type analyzed"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/comparison",
        help="Output directory for comparison results"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("SCENT Benchmark Comparison")
    print("="*60)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load benchmark statistics
    jaxqtl_stats_file = os.path.join(args.results_dir, "scent_jaxqtl", "benchmark_stats.json")
    r_stats_file = os.path.join(args.results_dir, "scent_r", "benchmark_stats.json")

    if os.path.exists(jaxqtl_stats_file):
        with open(jaxqtl_stats_file) as f:
            jaxqtl_stats = json.load(f)
    else:
        print(f"Warning: SCENT_jaxQTL stats not found: {jaxqtl_stats_file}")
        jaxqtl_stats = []

    if os.path.exists(r_stats_file):
        with open(r_stats_file) as f:
            r_stats = json.load(f)
    else:
        print(f"Warning: SCENT R stats not found: {r_stats_file}")
        r_stats = []

    # Compare performance
    if jaxqtl_stats and r_stats:
        plot_performance(jaxqtl_stats, r_stats, args.output_dir)

    # Compare results for each dataset
    all_comparison_stats = []

    for dataset in args.datasets:
        # Load results
        jaxqtl_results = load_results(
            args.results_dir, "scent_jaxqtl", dataset, args.cell_type
        )
        r_results = load_results(
            args.results_dir, "scent_r", dataset, args.cell_type
        )

        # Compare results
        if jaxqtl_results is not None and r_results is not None:
            comp_stats, both_results = compare_results(
                jaxqtl_results, r_results, dataset
            )

            if comp_stats is not None:
                all_comparison_stats.append(comp_stats)

                # Create comparison plots
                plot_comparisons(both_results, dataset, args.output_dir)

    # Save comparison statistics
    if all_comparison_stats:
        stats_file = os.path.join(args.output_dir, "comparison_stats.json")
        with open(stats_file, "w") as f:
            json.dump(all_comparison_stats, f, indent=2)
        print(f"\n✓ Comparison statistics saved to: {stats_file}")

        stats_df = pd.DataFrame(all_comparison_stats)
        csv_file = os.path.join(args.output_dir, "comparison_stats.csv")
        stats_df.to_csv(csv_file, index=False)
        print(f"✓ Comparison statistics saved to: {csv_file}")

    print("\n" + "="*60)
    print("✓ Comparison completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
