# -*- coding: utf-8 -*-
"""
plot_training_convergence.py
----------------------------
Reads results/training_metrics.csv and produces a 6-panel convergence
dashboard showing whether the DDQN agent learned correctly.

Panels
------
  1. Episode Reward Trend      -- should rise toward 0 from negative
  2. Hit Rate Trend            -- should rise toward 1.0
  3. Forwarded Count Trend     -- should fall as more requests served locally
  4. Violation (Drop) Trend    -- should fall as deadline management improves
  5. Avg Loss Trend            -- should fall and stabilise near 0
  6. Epsilon Trend             -- should decay from 1.0 to EPSILON_MIN

Each panel shows:
  - Raw values (light, semi-transparent)
  - Rolling mean (bold, window=10 episodes)
  - A horizontal reference line where applicable

Usage
-----
  # Focus on reward convergence (default)
  python plot_training_convergence.py --metric reward
  
  # Focus on hit_rate convergence
  python plot_training_convergence.py --metric hit_rate --csv results/training_metrics.csv
  
  # Focus on loss convergence with custom window
  python plot_training_convergence.py --metric avg_loss --window 20
  
  # Focus on violations (drops)
  python plot_training_convergence.py --metric violations --out results/conv_violations.png
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend -- safe for all environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_ROOT       = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CSV = os.path.join(_ROOT, "results", "training_metrics.csv")
_DEFAULT_OUT = os.path.join(_ROOT, "results", "convergence_dashboard.png")


# ---------------------------------------------------------------------------
# Convergence criteria (printed to console alongside the plot)
# ---------------------------------------------------------------------------

METRIC_CONFIGS = {
    "reward": {
        "title": "Episode Reward",
        "ylabel": "Reward",
        "color": "#1f77b4",
        "ref_val": 0.0,
        "ref_label": "Target >= 0",
        "convergence_type": "maximize",  # higher is better
        "threshold": -80.0,
        "threshold_op": "gt",  # greater than
        "improvement_threshold": 0.05,  # 5% improvement
        "note": "Should rise toward 0 (less negative)",
    },
    "hit_rate": {
        "title": "Cache Hit Rate",
        "ylabel": "Hit Rate",
        "color": "#2ca02c",
        "ref_val": 0.90,
        "ref_label": "Target >= 90%",
        "convergence_type": "maximize",
        "threshold": 0.88,
        "threshold_op": "gt",
        "improvement_threshold": 0.02,  # 2% improvement
        "note": "Should rise toward 1.0",
        "pct": True,
    },
    "forwarded": {
        "title": "Forwarded Request Count",
        "ylabel": "Requests forwarded",
        "color": "#ff7f0e",
        "ref_val": None,
        "ref_label": None,
        "convergence_type": "minimize",  # lower is better
        "threshold": None,
        "threshold_op": "trend_down",
        "improvement_threshold": 0.05,
        "note": "Should fall as local caching improves",
    },
    "violations": {
        "title": "SLA Violations (Drops)",
        "ylabel": "Dropped requests",
        "color": "#d62728",
        "ref_val": None,
        "ref_label": None,
        "convergence_type": "minimize",
        "threshold": None,
        "threshold_op": "trend_down",
        "improvement_threshold": 0.05,
        "note": "Should fall as deadline management improves",
    },
    "avg_loss": {
        "title": "Average Bellman Loss",
        "ylabel": "Huber loss",
        "color": "#9467bd",
        "ref_val": 0.001,
        "ref_label": "Target < 0.001",
        "convergence_type": "minimize",
        "threshold": 0.001,
        "threshold_op": "lt",
        "improvement_threshold": 0.10,
        "note": "Should decrease and stabilise",
        "log": True,
    },
    "epsilon": {
        "title": "Exploration Rate (Epsilon)",
        "ylabel": "Epsilon",
        "color": "#8c564b",
        "ref_val": 0.05,
        "ref_label": "Floor = 0.05",
        "convergence_type": "decay",
        "threshold": 0.06,
        "threshold_op": "lt",
        "improvement_threshold": None,
        "note": "Should decay from 1.0 to 0.05",
    },
}

def _rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def _trend_slope(series: pd.Series) -> float:
    """Return linear regression slope over the series."""
    x = np.arange(len(series), dtype=float)
    y = series.values.astype(float)
    if len(x) < 2:
        return 0.0
    p = np.polyfit(x, y, 1)
    return float(p[0])


def _check_primary_convergence(
    df: pd.DataFrame, 
    metric: str, 
    config: dict, 
    window: int
) -> dict:
    """
    Check convergence for the primary metric.
    
    Returns:
        dict with convergence status and detailed analysis
    """
    if metric not in df.columns:
        return {
            "converged": False,
            "reason": "Column not found in CSV",
            "confidence": 0.0,
        }
    
    series = df[metric].dropna()
    if len(series) < window * 2:
        return {
            "converged": False,
            "reason": f"Insufficient data (need >= {window * 2} episodes)",
            "confidence": 0.0,
        }
    
    # Get recent and previous values
    recent_window = series.iloc[-window:]
    previous_window = series.iloc[-2*window:-window]
    
    recent_avg = float(recent_window.mean())
    previous_avg = float(previous_window.mean())
    last_val = float(series.iloc[-1])
    
    # Calculate improvement
    if abs(previous_avg) > 1e-10:
        relative_improvement = abs(recent_avg - previous_avg) / abs(previous_avg)
    else:
        relative_improvement = 0.0
    
    # Calculate trend
    slope = _trend_slope(series)
    
    # Determine convergence based on metric type
    convergence_type = config["convergence_type"]
    threshold_op = config["threshold_op"]
    threshold = config.get("threshold")
    improvement_threshold = config.get("improvement_threshold", 0.05)
    
    checks = []
    
    # Check 1: Threshold check
    if threshold is not None:
        if threshold_op == "gt":
            threshold_met = last_val > threshold
            checks.append(("Threshold check", threshold_met, 
                          f"last={last_val:.4f} > {threshold}"))
        elif threshold_op == "lt":
            threshold_met = last_val < threshold
            checks.append(("Threshold check", threshold_met,
                          f"last={last_val:.6f} < {threshold}"))
    
    # Check 2: Trend check
    if threshold_op == "trend_down":
        trend_correct = slope < 0
        checks.append(("Trend check", trend_correct,
                      f"slope={slope:.6f} (decreasing)" if trend_correct else f"slope={slope:.6f} (not decreasing)"))
    elif convergence_type == "maximize":
        trend_correct = slope > 0 or relative_improvement < improvement_threshold
        checks.append(("Trend check", trend_correct,
                      f"slope={slope:.6f} (increasing or stable)" if trend_correct else f"slope={slope:.6f} (decreasing)"))
    elif convergence_type == "minimize":
        trend_correct = slope < 0 or relative_improvement < improvement_threshold
        checks.append(("Trend check", trend_correct,
                      f"slope={slope:.6f} (decreasing or stable)" if trend_correct else f"slope={slope:.6f} (increasing)"))
    
    # Check 3: Improvement plateau
    if improvement_threshold is not None:
        plateau_reached = relative_improvement < improvement_threshold
        checks.append(("Improvement check", plateau_reached,
                      f"improvement={relative_improvement*100:.2f}% < {improvement_threshold*100:.1f}%"))
    
    # Overall convergence decision
    passed_checks = sum(1 for _, passed, _ in checks if passed)
    total_checks = len(checks)
    confidence = passed_checks / total_checks if total_checks > 0 else 0.0
    
    converged = confidence >= 0.67  # At least 2/3 checks must pass
    
    return {
        "converged": converged,
        "recent_avg": recent_avg,
        "previous_avg": previous_avg,
        "last_val": last_val,
        "relative_improvement": relative_improvement,
        "slope": slope,
        "checks": checks,
        "confidence": confidence,
        "passed_checks": passed_checks,
        "total_checks": total_checks,
    }

def _estimate_episodes_needed(
    convergence_result: dict,
    current_episodes: int,
    metric_config: dict,
) -> dict:
    """Estimate additional episodes needed based on convergence status."""
    
    if convergence_result["converged"]:
        return {
            "needed": 0,
            "recommended": 0,
            "priority": "NONE",
            "reason": "Metric has converged",
        }
    
    improvement = convergence_result["relative_improvement"]
    
    # Estimate based on improvement rate
    if improvement > 0.10:  # Still improving significantly (>10%)
        needed = 1000
        priority = "HIGH"
        reason = f"Still improving significantly ({improvement*100:.1f}%)"
    elif improvement > 0.05:  # Moderate improvement (5-10%)
        needed = 500
        priority = "MEDIUM"
        reason = f"Moderate improvement detected ({improvement*100:.1f}%)"
    elif improvement > 0.02:  # Slow improvement (2-5%)
        needed = 300
        priority = "LOW"
        reason = f"Slow improvement detected ({improvement*100:.1f}%)"
    else:  # Minimal/no improvement
        needed = 200
        priority = "LOW"
        reason = "Minimal improvement, but not fully converged"
    
    # Check if minimum episode threshold met
    if current_episodes < 1000:
        min_needed = 1000 - current_episodes
        if min_needed > needed:
            needed = min_needed
            priority = "MEDIUM"
            reason += " (minimum episode threshold not met)"
    
    return {
        "needed": needed,
        "recommended": int(needed * 1.2),  # Add 20% buffer
        "priority": priority,
        "reason": reason,
    }

# ---------------------------------------------------------------------------
# Plot builder
# ---------------------------------------------------------------------------



def plot_convergence(
    csv_path: str = _DEFAULT_CSV,
    out_path: str = _DEFAULT_OUT,
    window: int = 10,
    primary_metric: str = "reward",
) -> None:

    if not os.path.exists(csv_path):
        print(f"ERROR: metrics file not found: {csv_path}")
        sys.exit(1)
    
    if primary_metric not in METRIC_CONFIGS:
        print(f"ERROR: Unknown metric '{primary_metric}'")
        print(f"Available metrics: {', '.join(METRIC_CONFIGS.keys())}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    episodes = df["episode"].values
    n_ep = len(episodes)

    print(f"  Loaded {n_ep} episodes from {csv_path}")
    print(f"  Primary convergence metric: {primary_metric}")

    primary_config = METRIC_CONFIGS[primary_metric]

    # ── Figure with focus on primary metric ────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
    
    # Large primary metric panel (top, spanning both columns)
    ax_primary = fig.add_subplot(gs[0, :])
    
    # Four smaller context panels (bottom two rows)
    context_metrics = [m for m in ["reward", "hit_rate", "avg_loss", "violations"] 
                      if m != primary_metric and m in df.columns][:4]
    ax_context = [fig.add_subplot(gs[i//2 + 1, i%2]) for i in range(len(context_metrics))]

    fig.suptitle(
        f"Training Convergence Analysis: {primary_config['title']}\n"
        f"{n_ep} Episodes  |  Window={window}  |  Config: {os.path.basename(csv_path)}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ── PRIMARY METRIC PANEL ────────────────────────────────────────
    if primary_metric in df.columns:
        raw = df[primary_metric].values.astype(float)
        smooth = _rolling(df[primary_metric], window).values

        # Raw line (light)
        ax_primary.plot(episodes, raw, color=primary_config["color"], alpha=0.25,
                       linewidth=1.0, label="Episode values")

        # Smoothed line (bold)
        ax_primary.plot(episodes, smooth, color=primary_config["color"], alpha=1.0,
                       linewidth=3.0, label=f"Rolling mean (window={window})")

        # Reference line
        if primary_config.get("ref_val") is not None:
            ax_primary.axhline(
                primary_config["ref_val"],
                color="black", linestyle="--",
                linewidth=1.5, label=primary_config["ref_label"], alpha=0.7,
            )

        # Log scale if needed
        if primary_config.get("log"):
            nonzero = raw[raw > 0]
            if len(nonzero) > 0:
                ax_primary.set_yscale("log")

        # Percent formatting if needed
        if primary_config.get("pct"):
            ax_primary.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

        # Convergence analysis
        conv_result = _check_primary_convergence(df, primary_metric, primary_config, window)
        
        # Annotate convergence status on plot
        status_text = "✓ CONVERGED" if conv_result["converged"] else "✗ NOT CONVERGED"
        status_color = "#2ca02c" if conv_result["converged"] else "#d62728"
        bbox_color = "lightgreen" if conv_result["converged"] else "lightcoral"
        
        info_text = (
            f"{status_text}\n"
            f"Recent avg: {conv_result['recent_avg']:.4f}\n"
            f"Improvement: {conv_result['relative_improvement']*100:+.2f}%\n"
            f"Confidence: {conv_result['confidence']*100:.0f}%\n"
            f"({conv_result['passed_checks']}/{conv_result['total_checks']} checks passed)"
        )
        
        ax_primary.text(
            0.02, 0.98, info_text, transform=ax_primary.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.8),
            color=status_color, fontweight='bold'
        )

        # Trend arrow
        slope = conv_result["slope"]
        last = conv_result["last_val"]
        arrow = "↓" if slope < 0 else ("↑" if slope > 0 else "→")
        
        ax_primary.annotate(
            f"  {arrow} {last:.4f}",
            xy=(episodes[-1], last),
            xytext=(episodes[-1] * 0.85, last),
            fontsize=10, color=status_color, fontweight="bold",
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor=status_color),
        )

        ax_primary.set_title(
            f"PRIMARY METRIC: {primary_config['title']}", 
            fontsize=13, fontweight="bold", pad=10
        )
        ax_primary.set_xlabel("Episode", fontsize=11, fontweight="bold")
        ax_primary.set_ylabel(primary_config["ylabel"], fontsize=11, fontweight="bold")
        ax_primary.legend(fontsize=10, loc="best")
        ax_primary.grid(True, alpha=0.3, linestyle=":")
        ax_primary.tick_params(labelsize=10)

        # Subtitle note
        ax_primary.text(
            0.98, 0.02, primary_config["note"], 
            transform=ax_primary.transAxes,
            fontsize=9, color="grey", ha="right", va="bottom", style="italic"
        )

    # ── CONTEXT METRICS PANELS ──────────────────────────────────────
    for ax, metric in zip(ax_context, context_metrics):
        if metric not in df.columns:
            continue
        
        cfg = METRIC_CONFIGS[metric]
        raw = df[metric].values.astype(float)
        smooth = _rolling(df[metric], window).values

        ax.plot(episodes, raw, color=cfg["color"], alpha=0.2, linewidth=0.5)
        ax.plot(episodes, smooth, color=cfg["color"], alpha=0.9, linewidth=1.5)

        if cfg.get("ref_val") is not None:
            ax.axhline(cfg["ref_val"], color="black", linestyle=":", 
                      linewidth=1.0, alpha=0.5)

        if cfg.get("log"):
            nonzero = raw[raw > 0]
            if len(nonzero) > 0:
                ax.set_yscale("log")
        
        if cfg.get("pct"):
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

        ax.set_title(cfg["title"], fontsize=9, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=8)
        ax.set_ylabel(cfg["ylabel"], fontsize=8)
        ax.grid(True, alpha=0.2, linestyle=":")
        ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved dashboard -> {out_path}")

    # ── CONVERGENCE REPORT ──────────────────────────────────────────
    _SEP = "=" * 80
    _DIV = "-" * 80
    
    print()
    print(_SEP)
    print(f"  CONVERGENCE ANALYSIS: {primary_config['title'].upper()}")
    print(_DIV)
    
    conv_result = _check_primary_convergence(df, primary_metric, primary_config, window)
    
    print(f"  Status: {'✓ CONVERGED' if conv_result['converged'] else '✗ NOT CONVERGED'}")
    print(f"  Confidence: {conv_result['confidence']*100:.1f}%")
    print()
    print(f"  Recent Average (last {window} ep): {conv_result['recent_avg']:.6f}")
    print(f"  Previous Average: {conv_result['previous_avg']:.6f}")
    print(f"  Current Value: {conv_result['last_val']:.6f}")
    print(f"  Relative Improvement: {conv_result['relative_improvement']*100:+.2f}%")
    print(f"  Trend Slope: {conv_result['slope']:.6f}")
    print()
    print("  Detailed Checks:")
    for check_name, passed, detail in conv_result["checks"]:
        mark = "✓ PASS" if passed else "✗ FAIL"
        print(f"    [{mark}] {check_name}: {detail}")
    
    print(_DIV)
    
    # Episode recommendation
    episodes_rec = _estimate_episodes_needed(conv_result, n_ep, primary_config)
    
    if episodes_rec["needed"] == 0:
        print("  ✓ TRAINING COMPLETE")
        print("  No additional episodes recommended.")
    else:
        print(f"  ⚠ ADDITIONAL TRAINING RECOMMENDED")
        print(f"  Priority: {episodes_rec['priority']}")
        print(f"  Reason: {episodes_rec['reason']}")
        print(f"  Minimum episodes needed: +{episodes_rec['needed']}")
        print(f"  Recommended episodes: +{episodes_rec['recommended']}")
        print(f"  New total: {n_ep + episodes_rec['recommended']} episodes")
    
    print(_SEP)

    # ── PHASE SUMMARY ───────────────────────────────────────────────
    q1 = n_ep // 4
    q3 = 3 * n_ep // 4

    print()
    print(f"  PHASE SUMMARY: {primary_config['title']}")
    print(_DIV)
    print(f"  {'Phase':<15} {'Episodes':>15}  {'Average':>18}")
    print(_DIV)

    if primary_metric in df.columns:
        s = df[primary_metric].values.astype(float)
        early = s[:q1].mean()
        mid = s[q1:q3].mean()
        late = s[q3:].mean()
        
        fmt = "{:>+.6f}" if not primary_config.get("pct") else "{:>.2%}"
        
        print(f"  {'Early':<15} {'1-'+str(q1):>15}  {fmt.format(early):>18}")
        print(f"  {'Mid':<15} {str(q1)+'-'+str(q3):>15}  {fmt.format(mid):>18}")
        print(f"  {'Late':<15} {str(q3)+'-'+str(n_ep):>15}  {fmt.format(late):>18}")
        
        # Show improvement
        early_to_late = ((late - early) / abs(early)) if abs(early) > 1e-10 else 0
        print(_DIV)
        print(f"  Early → Late improvement: {early_to_late*100:+.2f}%")

    print(_SEP)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Plot DDQN training convergence from training_metrics.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python plot_training_convergence.py\n"
            "  python plot_training_convergence.py --csv results/training_metrics.csv\n"
            "  python plot_training_convergence.py --window 20 --out results/conv.png\n"
        ),
    )
    p.add_argument("--csv",    type=str, default=_DEFAULT_CSV,
                   help=f"Path to training_metrics.csv (default: {_DEFAULT_CSV})")
    p.add_argument("--out",    type=str, default=_DEFAULT_OUT,
                   help=f"Output PNG path (default: {_DEFAULT_OUT})")
    p.add_argument("--window", type=int, default=10,
                   help="Rolling mean window in episodes (default: 10)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print("=" * 60)
    print("  DDQN Training Convergence Plotter")
    print("=" * 60)
    plot_convergence(
        csv_path = args.csv,
        out_path = args.out,
        window   = args.window,
    )
