"""
# -*- coding: utf-8 -*-
plot_individual_training_metrics.py
-----------------------------------
Generates separate high-resolution graphs for individual training metrics
from the training CSV file. Each metric gets its own dedicated chart
following IEEE standards for publication quality.

Metrics Generated:
------------------
1. Average Bellman Loss
2. Exploration Rate (Epsilon)
3. Cache Hit Rate Trend
4. Episode Reward Trend

Usage:
------
  python plot_individual_training_metrics.py
  python plot_individual_training_metrics.py --csv results/finetune_metrics.csv
  python plot_individual_training_metrics.py --csv results/training_metrics.csv --out-dir results/individual_plots
  python plot_individual_training_metrics.py --window 20  # wider smoothing
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# IEEE standard plot settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linewidth'] = 0.8

_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CSV = os.path.join(_ROOT, "results", "training_metrics.csv")
_DEFAULT_OUT_DIR = os.path.join(_ROOT, "results", "training_plots")

# Chart settings
CHART_WIDTH = 15
CHART_HEIGHT = 10
CHART_DPI = 300


def _rolling(series: pd.Series, window: int) -> pd.Series:
    """Calculate rolling mean for smoothing."""
    return series.rolling(window=window, min_periods=1).mean()


def _trend_slope(series: pd.Series) -> float:
    """Return linear regression slope over the series."""
    x = np.arange(len(series), dtype=float)
    y = series.values.astype(float)
    if len(x) < 2:
        return 0.0
    p = np.polyfit(x, y, 1)
    return float(p[0])


def plot_bellman_loss(df: pd.DataFrame, out_dir: str, window: int = 10):
    """Plot Average Bellman Loss trend."""

    if "avg_loss" not in df.columns:
        print("  WARNING: 'avg_loss' column not found. Skipping.")
        return

    episodes = df["episode"].values
    raw = df["avg_loss"].values.astype(float)
    smooth = _rolling(df["avg_loss"], window).values

    fig, ax = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))

    # Plot raw data (light)
    ax.plot(episodes, raw, color='#9467bd', alpha=0.25, linewidth=1.5, 
            label='Raw Loss', zorder=1)

    # Plot smoothed data (bold)
    ax.plot(episodes, smooth, color='#9467bd', alpha=0.95, linewidth=4.0, 
            label=f'Rolling Mean (w={window})', zorder=2)

    # Reference line
    ax.axhline(0.001, color='red', linestyle='--', linewidth=2.0, 
               label='Target < 0.001', alpha=0.7, zorder=3)

    # Annotations
    last_val = smooth[-1]
    slope = _trend_slope(df["avg_loss"])
    trend_text = "Decreasing" if slope < 0 else "Increasing" if slope > 0 else "Stable"

    ax.text(0.02, 0.98, f'Final Loss: {last_val:.6f}\nTrend: {trend_text}\nSlope: {slope:.8f}',
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.5))

    # Use log scale for better visualization
    nonzero = raw[raw > 0]
    if len(nonzero) > 0:
        ax.set_yscale('log')
        ax.set_ylabel("Average Bellman Loss (log scale)", fontsize=25, fontweight="bold", labelpad=10)
    else:
        ax.set_ylabel("Average Bellman Loss", fontsize=25, fontweight="bold", labelpad=10)

    ax.set_xlabel("Episode", fontsize=25, fontweight="bold", labelpad=12)
    ax.set_title("SPRIHA Training: Average Bellman Loss Convergence", 
                fontsize=22, fontweight="bold", pad=23)
    ax.legend(fontsize=18, loc="upper right", framealpha=0.95, edgecolor='black', frameon=True)
    ax.grid(True, alpha=0.35, linestyle='--', linewidth=1.0)
    ax.tick_params(axis='both', labelsize=25, width=1.8, length=7, pad=6)
    ax.tick_params(axis='y', rotation=45)
    ax.set_xlim(left=0)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "bellman_loss_trend.png")
    plt.savefig(out_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_epsilon_decay(df: pd.DataFrame, out_dir: str, window: int = 10):
    """Plot Exploration Rate (Epsilon) decay."""

    if "epsilon" not in df.columns:
        print("  WARNING: 'epsilon' column not found. Skipping.")
        return

    episodes = df["episode"].values
    raw = df["epsilon"].values.astype(float)
    smooth = _rolling(df["epsilon"], window).values

    fig, ax = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))

    # Plot raw data (light)
    ax.plot(episodes, raw, color='#8c564b', alpha=0.25, linewidth=1.5, 
            label='Raw Epsilon', zorder=1)

    # Plot smoothed data (bold)
    ax.plot(episodes, smooth, color='#8c564b', alpha=0.95, linewidth=4.0, 
            label=f'Rolling Mean (w={window})', zorder=2)

    # Reference lines
    ax.axhline(1.0, color='green', linestyle=':', linewidth=2.0, 
               label='Initial ε = 1.0', alpha=0.7, zorder=3)
    ax.axhline(0.01, color='red', linestyle='--', linewidth=2.0, 
               label='Floor ε = 0.01', alpha=0.7, zorder=3)

    # Annotations
    last_val = smooth[-1]
    first_val = smooth[0]
    decay_pct = ((first_val - last_val) / first_val) * 100

    ax.text(0.02, 0.98, f'Initial ε: {first_val:.4f}\nFinal ε: {last_val:.4f}\nDecay: {decay_pct:.1f}%',
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.5))

    ax.set_xlabel("Episode", fontsize=25, fontweight="bold", labelpad=12)
    ax.set_ylabel("Exploration Rate (Epsilon)", fontsize=25, fontweight="bold", labelpad=10)
    ax.set_title("SPRIHA Training: Epsilon Decay Schedule", 
                fontsize=22, fontweight="bold", pad=23)
    ax.legend(fontsize=18, loc="upper right", framealpha=0.95, edgecolor='black', frameon=True)
    ax.grid(True, alpha=0.35, linestyle='--', linewidth=1.0)
    ax.tick_params(axis='both', labelsize=25, width=1.8, length=7, pad=6)
    ax.tick_params(axis='y', rotation=45)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=1.1)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "epsilon_decay_trend.png")
    plt.savefig(out_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_hit_rate(df: pd.DataFrame, out_dir: str, window: int = 10):
    """Plot Cache Hit Rate trend."""

    if "hit_rate" not in df.columns:
        print("  WARNING: 'hit_rate' column not found. Skipping.")
        return

    episodes = df["episode"].values
    raw = df["hit_rate"].values.astype(float)
    smooth = _rolling(df["hit_rate"], window).values

    fig, ax = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))

    # Plot raw data (light)
    ax.plot(episodes, raw, color='#2ca02c', alpha=0.25, linewidth=1.5, 
            label='Raw Hit Rate', zorder=1)

    # Plot smoothed data (bold)
    ax.plot(episodes, smooth, color='#2ca02c', alpha=0.95, linewidth=4.0, 
            label=f'Rolling Mean (w={window})', zorder=2)

    # Reference line
    ax.axhline(0.90, color='red', linestyle='--', linewidth=2.0, 
               label='Target >= 90%', alpha=0.7, zorder=3)

    # Fill area under curve
    ax.fill_between(episodes, smooth, alpha=0.15, color='#2ca02c', zorder=0)

    # Annotations
    last_val = smooth[-1]
    first_val = smooth[0]
    improvement = ((last_val - first_val) / first_val) * 100
    slope = _trend_slope(df["hit_rate"])

    ax.text(0.02, 0.02, f'Initial: {first_val*100:.1f}%\nFinal: {last_val*100:.1f}%\nImprovement: +{improvement:.1f}%',
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            verticalalignment='bottom', bbox=dict(boxstyle='round', 
            facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.5))

    # Convergence indicator
    if last_val >= 0.90:
        convergence_text = "[PASS] Target Achieved"
        conv_color = 'green'
    else:
        convergence_text = "[WARN] Below Target"
        conv_color = 'orange'

    ax.text(0.98, 0.98, convergence_text,
            transform=ax.transAxes, fontsize=16, fontweight='bold',
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=conv_color, 
            alpha=0.3, edgecolor='black', linewidth=1.5))

    ax.set_xlabel("Episode", fontsize=25, fontweight="bold", labelpad=12)
    ax.set_ylabel("Cache Hit Rate", fontsize=25, fontweight="bold", labelpad=10)
    ax.set_title("SPRIHA Training: Cache Hit Rate Improvement", 
                fontsize=22, fontweight="bold", pad=23)
    ax.legend(fontsize=18, loc="lower right", framealpha=0.95, edgecolor='black', frameon=True)
    ax.grid(True, alpha=0.35, linestyle='--', linewidth=1.0)
    ax.tick_params(axis='both', labelsize=25, width=1.8, length=7, pad=6)
    ax.tick_params(axis='y', rotation=45)

    # Format as percentage
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=1.05)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "hit_rate_trend.png")
    plt.savefig(out_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_episode_reward(df: pd.DataFrame, out_dir: str, window: int = 10):
    """Plot Episode Reward trend."""

    if "reward" not in df.columns:
        print("  WARNING: 'reward' column not found. Skipping.")
        return

    episodes = df["episode"].values
    raw = df["reward"].values.astype(float)
    smooth = _rolling(df["reward"], window).values

    fig, ax = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))

    # Plot raw data (light)
    ax.plot(episodes, raw, color='#1f77b4', alpha=0.25, linewidth=1.5, 
            label='Raw Reward', zorder=1)

    # Plot smoothed data (bold)
    ax.plot(episodes, smooth, color='#1f77b4', alpha=0.95, linewidth=4.0, 
            label=f'Rolling Mean (w={window})', zorder=2)

    # Reference line
    ax.axhline(0, color='green', linestyle='--', linewidth=2.0, 
               label='Target >= 0', alpha=0.7, zorder=3)

    # Fill area under curve (different colors for positive/negative)
    ax.fill_between(episodes, smooth, 0, where=(smooth >= 0), 
                    alpha=0.15, color='green', interpolate=True, zorder=0)
    ax.fill_between(episodes, smooth, 0, where=(smooth < 0), 
                    alpha=0.15, color='red', interpolate=True, zorder=0)

    # Annotations
    last_val = smooth[-1]
    first_val = smooth[0]
    improvement = last_val - first_val
    slope = _trend_slope(df["reward"])
    trend_text = "Improving" if slope > 0 else "Degrading" if slope < 0 else "Stable"

    ax.text(0.02, 0.02, f'Initial: {first_val:.1f}\nFinal: {last_val:.1f}\nImprovement: {improvement:+.1f}',
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            verticalalignment='bottom', bbox=dict(boxstyle='round', 
            facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.5))

    # Convergence indicator
    if last_val >= -80:
        convergence_text = "[PASS] Good Performance"
        conv_color = 'green'
    elif last_val >= -150:
        convergence_text = "[WARN] Moderate Performance"
        conv_color = 'yellow'
    else:
        convergence_text = "[FAIL] Needs Improvement"
        conv_color = 'red'

    ax.text(0.98, 0.98, f'{convergence_text}\nTrend: {trend_text}',
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=conv_color, 
            alpha=0.3, edgecolor='black', linewidth=1.5))

    ax.set_xlabel("Episode", fontsize=25, fontweight="bold", labelpad=12)
    ax.set_ylabel("Episode Reward", fontsize=25, fontweight="bold", labelpad=10)
    ax.set_title("SPRIHA Training: Episode Reward Progression", 
                fontsize=22, fontweight="bold", pad=23)
    ax.legend(fontsize=18, loc="lower right", framealpha=0.95, edgecolor='black', frameon=True)
    ax.grid(True, alpha=0.35, linestyle='--', linewidth=1.0)
    ax.tick_params(axis='both', labelsize=25, width=1.8, length=7, pad=6)
    ax.tick_params(axis='y', rotation=45)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    # Format y-axis with thousand separator
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    plt.tight_layout()
    out_path = os.path.join(out_dir, "episode_reward_trend.png")
    plt.savefig(out_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate individual training metric plots for SPRIHA")
    parser.add_argument("--csv", default=_DEFAULT_CSV, 
                       help="Path to training metrics CSV")
    parser.add_argument("--out-dir", default=_DEFAULT_OUT_DIR, 
                       help="Output directory for plots")
    parser.add_argument("--window", type=int, default=10, 
                       help="Rolling mean window size")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: CSV file not found: {args.csv}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    print(f"\nLoading training data from: {args.csv}")
    df = pd.read_csv(args.csv)
    n_episodes = len(df)
    print(f"  Found {n_episodes} episodes")

    # Generate plots
    print("\nGenerating individual metric plots...")
    print("=" * 60)

    plot_bellman_loss(df, args.out_dir, args.window)
    plot_epsilon_decay(df, args.out_dir, args.window)
    plot_hit_rate(df, args.out_dir, args.window)
    plot_episode_reward(df, args.out_dir, args.window)

    print("=" * 60)
    print(f"\n[SUCCESS] Complete! All plots saved to: {args.out_dir}")
    print("\nGenerated files:")
    print("  1. bellman_loss_trend.png     - Average Bellman Loss")
    print("  2. epsilon_decay_trend.png     - Exploration Rate (Epsilon)")
    print("  3. hit_rate_trend.png          - Cache Hit Rate")
    print("  4. episode_reward_trend.png    - Episode Reward")
    print("\nAll charts: 300 DPI, 15x10 inches, IEEE standard formatting")
    print()


if __name__ == "__main__":
    main()
