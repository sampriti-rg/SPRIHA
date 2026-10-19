"""
Plot comparison charts for different tree structures (3-2, 3-3, 3-4, 3-5).

Usage:
    python plot_tree_comparison.py
    python plot_tree_comparison.py --metric drop_rate --output results/comparison_drop_rate.png
    python plot_tree_comparison.py --metric all --output results/tree_comparison_all.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Default file paths
DEFAULT_FILES = {
    "3-2": "results/policy_comparison_v3-2tree.csv",
    "3-3": "results/policy_comparison_v3-3tree.csv",
    "3-4": "results/policy_comparison_v3-4tree.csv",
    "3-5": "results/policy_comparison_v3-5tree.csv",
}

# Metrics to plot
METRICS = {
    "drop_rate": {"title": "Drop Rate Comparison", "ylabel": "Drop Rate (%)", "scale": 100},
    "local_rate": {"title": "Local Execution Rate Comparison", "ylabel": "Local Execution Rate (%)", "scale": 100},
    "cloud_rate": {"title": "Cloud Escalation Rate Comparison", "ylabel": "Cloud Escalation Rate (%)", "scale": 100},
    "total_revenue": {"title": "Total Revenue Comparison", "ylabel": "Total Revenue", "scale": 1}
    }

    # "hit_rate_mean": {"title": "Hit Rate Comparison", "ylabel": "Hit Rate (%)", "scale": 100},
    # "rev_per_request": {"title": "Revenue per Request Comparison", "ylabel": "Revenue per Request", "scale": 1},



def load_data(file_paths):
    """Load data from multiple CSV files."""
    data = {}
    for tree_type, filepath in file_paths.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            data[tree_type] = df
            print(f"✓ Loaded {filepath}: {len(df)} rows")
        else:
            print(f"⚠ File not found: {filepath}")
    return data


def plot_metric(data, metric_name, ax=None, show_legend=True):
    """Plot a single metric across different tree structures."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_info = METRICS[metric_name]
    scale = metric_info["scale"]
    
    # Define colors and markers for each tree type
    styles = {
        "3-2": {"color": "blue", "marker": "o", "label": "3 Levels, 2 Children (8 leaves)"},
        "3-3": {"color": "green", "marker": "s", "label": "3 Levels, 3 Children (27 leaves)"},
        "3-4": {"color": "red", "marker": "^", "label": "3 Levels, 4 Children (64 leaves)"},
        "3-5": {"color": "purple", "marker": "D", "label": "3 Levels, 5 Children (125 leaves)"},
    }
    
    for tree_type, df in sorted(data.items()):
        if metric_name in df.columns:
            style = styles.get(tree_type, {"color": "gray", "marker": "x", "label": tree_type})
            ax.plot(
                df["n_requests"],
                df[metric_name] * scale,
                marker=style["marker"],
                color=style["color"],
                label=style["label"],
                linewidth=2,
                markersize=8
            )
    
    ax.set_xlabel("Number of Requests", fontsize=12, fontweight="bold")
    ax.set_ylabel(metric_info["ylabel"], fontsize=12, fontweight="bold")
    ax.set_title(metric_info["title"], fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    if show_legend:
        ax.legend(loc="best", fontsize=10)
    
    return ax


def plot_all_metrics(data, output_path):
    """Create a multi-panel figure with all metrics."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle("Tree Structure Comparison - DDQN Policy Performance", 
                 fontsize=16, fontweight="bold", y=0.995)
    
    metrics_list = list(METRICS.keys())
    
    for idx, metric in enumerate(metrics_list):
        row = idx // 2
        col = idx % 2
        plot_metric(data, metric, ax=axes[row, col], show_legend=(idx == 0))
    
    plt.tight_layout()
    
    # Save figure
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved comprehensive comparison chart: {output_file}")
    
    return fig


def plot_single_metric(data, metric_name, output_path):
    """Create a single metric comparison chart."""
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_metric(data, metric_name, ax=ax, show_legend=True)
    
    plt.tight_layout()
    
    # Save figure
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved {metric_name} comparison chart: {output_file}")
    
    return fig


def generate_summary_table(data):
    """Generate a summary comparison table for the maximum load level."""
    print("\n" + "="*80)
    print("SUMMARY TABLE - Performance at Maximum Load")
    print("="*80)
    
    summary_data = []
    
    for tree_type, df in sorted(data.items()):
        if len(df) > 0:
            last_row = df.iloc[-1]
            summary_data.append({
                "Tree Type": tree_type,
                "Max Requests": int(last_row["n_requests"]),
                "Drop Rate (%)": f"{last_row['drop_rate']*100:.2f}",
                "Local Rate (%)": f"{last_row['local_rate']*100:.2f}",
                "Cloud Rate (%)": f"{last_row['cloud_rate']*100:.2f}",
                "Hit Rate (%)": f"{last_row['hit_rate_mean']*100:.2f}",
                "Total Revenue": f"{last_row['total_revenue']:.2f}",
                "Rev/Request": f"{last_row['rev_per_request']:.4f}",
                "Executed": int(last_row["executed"]),
                "Dropped": int(last_row["dropped"]),
            })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("="*80 + "\n")
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Compare DDQN performance across different tree structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot all metrics
  python plot_tree_comparison.py --metric all --output results/tree_comparison_all.png
  
  # Plot drop rate only
  python plot_tree_comparison.py --metric drop_rate --output results/comparison_drop_rate.png
  
  # Plot revenue comparison
  python plot_tree_comparison.py --metric total_revenue --output results/comparison_revenue.png
        """
    )
    
    parser.add_argument(
        "--metric", type=str, default="all",
        choices=list(METRICS.keys()) + ["all"],
        help="Metric to plot (default: all)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="results/tree_structure_comparison.png",
        help="Output image path (default: results/tree_structure_comparison.png)"
    )
    parser.add_argument(
        "--files", type=str, nargs="+",
        help="Custom file paths (format: tree_type:path, e.g., 3-2:results/file.csv)"
    )
    parser.add_argument(
        "--show-summary", action="store_true", default=True,
        help="Show summary table (default: True)"
    )
    
    args = parser.parse_args()
    
    # Load file paths
    file_paths = DEFAULT_FILES.copy()
    
    if args.files:
        for file_spec in args.files:
            if ":" in file_spec:
                tree_type, path = file_spec.split(":", 1)
                file_paths[tree_type] = path
    
    print("="*80)
    print("Tree Structure Comparison - DDQN Policy")
    print("="*80)
    
    # Load data
    data = load_data(file_paths)
    
    if not data:
        print("\n✗ No data files found!")
        return
    
    # Generate summary table
    if args.show_summary:
        generate_summary_table(data)
    
    # Plot
    if args.metric == "all":
        plot_all_metrics(data, args.output)
    else:
        plot_single_metric(data, args.metric, args.output)
    
    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    main()
