#!/usr/bin/env python3
"""
Analysis script for FlexMoRE model results.
Analyzes the impact of low-rank decomposition rank on benchmark performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def parse_results_file(filepath):
    """Parse the results CSV file into a structured DataFrame."""
    data = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split model name from scores
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue

            model_name = parts[0]
            scores_str = parts[1]

            # Extract rank from model name
            rank_match = re.search(r'-r(\d+)', model_name)
            rank = int(rank_match.group(1)) if rank_match else (131072 if 'best2' in model_name else (65536 if 'best' in model_name else 32768))
            if rank is None:
                print(f"Warning: Could not determine rank for model '{model_name}'")

            # Extract base model type
            if 'Flex-code' in model_name:
                model_type = 'Flex-code'
            elif 'Flex-creative' in model_name:
                model_type = 'Flex-creative'
            elif 'Flex-news' in model_name:
                model_type = 'Flex-news'
            elif 'Flex-reddit' in model_name:
                model_type = 'Flex-reddit'
            elif 'Flex-pes2o' in model_name:
                model_type = 'Flex-pes2o'
            elif 'Flex-math' in model_name:
                model_type = 'Flex-math'
            elif 'FlexOlmo' in model_name:
                if '-a2' in model_name:
                    model_type = 'FlexOlmo-a2'
                elif '-a4' in model_name:
                    model_type = 'FlexOlmo-a4'
                elif '-a7' in model_name:
                    model_type = 'FlexOlmo-a7'
                else:
                    model_type = 'FlexOlmo-unknown'
            else:
                model_type = 'Unknown'

            # Parse benchmark scores
            row = {'model_name': model_name, 'model_type': model_type, 'rank': rank}

            # Extract all benchmark scores
            benchmark_pattern = r'(\w+):\s+([\d.]+)'
            for match in re.finditer(benchmark_pattern, scores_str):
                benchmark_name = match.group(1)
                score = float(match.group(2))
                row[benchmark_name] = score

            data.append(row)

    return pd.DataFrame(data)

def calculate_statistics(df):
    """Calculate statistics across benchmarks for each model."""
    benchmark_cols = [col for col in df.columns if col not in ['model_name', 'model_type', 'rank']]

    df['mean_score'] = df[benchmark_cols].mean(axis=1)
    df['std_score'] = df[benchmark_cols].std(axis=1)
    df['min_score'] = df[benchmark_cols].min(axis=1)
    df['max_score'] = df[benchmark_cols].max(axis=1)

    return df

def plot_rank_vs_performance(df, output_dir):
    """Plot how rank affects overall performance for each model type."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    model_types = df['model_type'].unique()

    for idx, model_type in enumerate(model_types[:9]):
        if idx >= len(axes):
            break

        subset = df[df['model_type'] == model_type].copy()
        subset = subset.dropna(subset=['rank'])
        subset = subset.sort_values('rank')

        ax = axes[idx]
        ax.plot(subset['rank'], subset['mean_score'], marker='o', linewidth=2, markersize=8, label='Mean Score')
        ax.fill_between(subset['rank'],
                        subset['mean_score'] - subset['std_score'],
                        subset['mean_score'] + subset['std_score'],
                        alpha=0.3)

        ax.set_xscale('log', base=2)
        ax.set_xlabel('Rank (log scale)', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0.55, 0.775)
        ax.set_title(f'{model_type}: Rank vs Performance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'rank_vs_performance.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'rank_vs_performance.png'}")
    plt.close()

def plot_benchmark_heatmap(df, output_dir):
    """Create heatmap of benchmark scores across different ranks for each model type."""
    benchmark_cols = [col for col in df.columns if col not in ['model_name', 'model_type', 'rank',
                                                                 'mean_score', 'std_score', 'min_score', 'max_score']]

    for model_type in df['model_type'].unique():
        subset = df[df['model_type'] == model_type].copy()
        subset = subset.dropna(subset=['rank'])
        subset = subset.sort_values('rank')

        if len(subset) < 2:
            continue

        # Create pivot table
        pivot_data = subset.set_index('rank')[benchmark_cols].T

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn',
                   vmin=0.4, vmax=0.9, ax=ax, cbar_kws={'label': 'Score'})
        ax.set_title(f'{model_type}: Benchmark Scores by Rank', fontsize=16, fontweight='bold')
        ax.set_xlabel('Rank', fontsize=12)
        ax.set_ylabel('Benchmark', fontsize=12)

        plt.tight_layout()
        plt.savefig(output_dir / f'heatmap_{model_type.lower()}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / f'heatmap_{model_type.lower()}.png'}")
        plt.close()

def plot_benchmark_comparison(df, output_dir):
    """Compare specific benchmarks across different ranks."""
    benchmark_cols = [col for col in df.columns if col not in ['model_name', 'model_type', 'rank',
                                                                 'mean_score', 'std_score', 'min_score', 'max_score']]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    for idx, benchmark in enumerate(benchmark_cols[:9]):
        ax = axes[idx]

        for model_type in df['model_type'].unique():
            subset = df[df['model_type'] == model_type].copy()
            subset = subset.dropna(subset=['rank'])
            subset = subset.sort_values('rank')

            if len(subset) > 1:
                ax.plot(subset['rank'], subset[benchmark], marker='o', label=model_type, alpha=0.7)

        ax.set_xscale('log', base=2)
        ax.set_xlabel('Rank (log scale)', fontsize=10)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title(benchmark, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'benchmark_comparison.png'}")
    plt.close()

def find_optimal_ranks(df):
    """Find optimal rank for each model type based on mean score."""
    results = []

    for model_type in df['model_type'].unique():
        subset = df[df['model_type'] == model_type].copy()
        subset = subset.dropna(subset=['rank'])

        if len(subset) == 0:
            continue

        # Find rank with best mean score
        best_idx = subset['mean_score'].idxmax()
        best_row = subset.loc[best_idx]

        best_rank = best_row['rank']
        if best_rank == 131072:
            best_rank_label = 'best2'
        elif best_rank == 65536:
            best_rank_label = 'best'
        elif best_rank == 32768:
            best_rank_label = 'full'
        else:
            best_rank_label = str(int(best_rank))
        results.append({
            'model_type': model_type,
            'optimal_rank': best_rank,
            'mean_score': best_row['mean_score'],
            'std_score': best_row['std_score']
        })

    return pd.DataFrame(results)

def generate_summary_report(df, optimal_ranks, output_dir):
    """Generate a text summary report."""
    report_path = output_dir / 'analysis_report.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FlexMoRE Low-Rank Decomposition Analysis Report\n")
        f.write("=" * 80 + "\n\n")

        f.write("OPTIMAL RANKS BY MODEL TYPE\n")
        f.write("-" * 80 + "\n")
        for _, row in optimal_ranks.iterrows():
            f.write(f"{row['model_type']:20s} | Optimal Rank: {int(row['optimal_rank']):5d} | "
                   f"Mean Score: {row['mean_score']:.4f} ± {row['std_score']:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("RANK RANGE ANALYSIS\n")
        f.write("-" * 80 + "\n")

        for model_type in df['model_type'].unique():
            subset = df[df['model_type'] == model_type].copy()
            subset = subset.dropna(subset=['rank'])

            if len(subset) == 0:
                continue

            f.write(f"\n{model_type}:\n")
            f.write(f"  Ranks tested: {sorted(subset['rank'].unique())}\n")
            f.write(f"  Score range: {subset['mean_score'].min():.4f} - {subset['mean_score'].max():.4f}\n")
            f.write(f"  Score variance: {subset['mean_score'].var():.6f}\n")

            # Check for diminishing returns
            subset_sorted = subset.sort_values('rank')
            if len(subset_sorted) > 1:
                score_changes = subset_sorted['mean_score'].diff()
                f.write(f"  Average score change per rank increase: {score_changes.mean():.6f}\n")

    print(f"Saved: {report_path}")

def main():
    """Main analysis pipeline."""
    import argparse
    parser = argparse.ArgumentParser(description="Analyze FlexMoRE model results.")
    parser.add_argument('resultsfile', type=str, default=None,
                        help="Path to the results file.")
    parser.add_argument('-o', '--output-dir', type=str, default='analysis_output',
                        help="Path for output directory")
    args = parser.parse_args()
    # Setup paths
    results_file = Path(args.resultsfile)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("FlexMoRE Results Analysis")
    print("=" * 80)

    # Load and parse data
    print("\n1. Loading data...")
    df = parse_results_file(results_file)
    print(f"   Loaded {len(df)} model configurations")

    # Calculate statistics
    print("\n2. Calculating statistics...")
    df = calculate_statistics(df)

    # Save processed data
    processed_path = output_dir / 'processed_results.csv'
    df.to_csv(processed_path, index=False)
    print(f"   Saved processed data: {processed_path}")

    # Find optimal ranks
    print("\n3. Finding optimal ranks...")
    optimal_ranks = find_optimal_ranks(df)
    print(optimal_ranks.to_string(index=False))

    # Generate visualizations
    print("\n4. Generating visualizations...")
    plot_rank_vs_performance(df, output_dir)
    plot_benchmark_heatmap(df, output_dir)
    plot_benchmark_comparison(df, output_dir)

    # Generate report
    print("\n5. Generating summary report...")
    generate_summary_report(df, optimal_ranks, output_dir)

    print("\n" + "=" * 80)
    print(f"Analysis complete! Check the '{args.output_dir}' directory for results.")
    print("=" * 80)

if __name__ == '__main__':
    main()
