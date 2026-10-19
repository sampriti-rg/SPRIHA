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

CONVERGENCE_CHECKS = [
    # (column, direction, threshold, description)
    ("reward",     "last_gt",  -80.0,   "Final episode reward > -80"),
    ("hit_rate",   "last_gt",   0.88,   "Final hit rate > 88 %"),
    ("forwarded",  "trend_dn",  None,   "Forwarded count trending down"),
    ("violations", "trend_dn",  None,   "Violations trending down"),
    ("avg_loss",   "last_lt",   0.001,  "Final avg loss < 0.001"),
    ("epsilon",    "last_lt",   0.06,   "Epsilon decayed to near floor"),
]


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


def _check_convergence(df: pd.DataFrame) -> list:
    results = []
    for col, direction, threshold, desc in CONVERGENCE_CHECKS:
        if col not in df.columns:
            results.append((desc, False, "column missing"))
            continue
        series = df[col].dropna()
        if len(series) == 0:
            results.append((desc, False, "no data"))
            continue

        last_val = float(series.iloc[-1])

        if direction == "last_gt":
            passed = last_val > threshold
            detail = f"last={last_val:.4f}, threshold>{threshold}"
        elif direction == "last_lt":
            passed = last_val < threshold
            detail = f"last={last_val:.6f}, threshold<{threshold}"
        elif direction == "trend_dn":
            slope  = _trend_slope(series)
            passed = slope < 0
            detail = f"slope={slope:.4f} ({'decreasing' if passed else 'not decreasing'})"
        else:
            passed = False
            detail = "unknown check"

        results.append((desc, passed, detail))
    return results


# ---------------------------------------------------------------------------
# Plot builder
# ---------------------------------------------------------------------------

def plot_convergence(
    csv_path : str = _DEFAULT_CSV,
    out_path : str = _DEFAULT_OUT,
    window   : int = 10,
) -> None:

    if not os.path.exists(csv_path):
        print(f"ERROR: metrics file not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    episodes = df["episode"].values
    n_ep     = len(episodes)

    print(f"  Loaded {n_ep} episodes from {csv_path}")

    # ── Panel definitions ────────────────────────────────────────────
    panels = [
        {
            "col"      : "reward",
            "title"    : "Episode Reward",
            "ylabel"   : "Reward",
            "color"    : "#1f77b4",
            "ref_val"  : 0.0,
            "ref_label": "Target >= 0",
            "ref_style": "--",
        "invert"   : False,
            "note"     : "Should rise toward 0 (less negative)",
        },
        {
            "col"      : "hit_rate",
            "title"    : "Cache Hit Rate",
            "ylabel"   : "Hit Rate",
            "color"    : "#2ca02c",
            "ref_val"  : 0.90,
            "ref_label": "Target >= 90%",
            "ref_style": "--",
            "invert"   : False,
            "note"     : "Should rise toward 1.0",
            "pct"      : True,
        },
        {
            "col"      : "forwarded",
            "title"    : "Forwarded Request Count",
            "ylabel"   : "Requests forwarded",
            "color"    : "#ff7f0e",
            "ref_val"  : None,
            "ref_label": None,
            "ref_style": "--",
            "invert"   : True,
            "note"     : "Should fall as local caching improves",
        },
        {
            "col"      : "violations",
            "title"    : "SLA Violations (Drops)",
            "ylabel"   : "Dropped requests",
            "color"    : "#d62728",
            "ref_val"  : None,
            "ref_label": None,
            "ref_style": "--",
            "invert"   : True,
            "note"     : "Should fall as deadline management improves",
        },
        {
            "col"      : "avg_loss",
            "title"    : "Average Bellman Loss",
            "ylabel"   : "Huber loss",
            "color"    : "#9467bd",
            "ref_val"  : 0.001,
            "ref_label": "Target < 0.001",
            "ref_style": ":",
            "invert"   : True,
            "note"     : "Should decrease and stabilise",
            "log"      : True,
        },
        {
            "col"      : "epsilon",
            "title"    : "Exploration Rate (Epsilon)",
            "ylabel"   : "Epsilon",
            "color"    : "#8c564b",
            "ref_val"  : 0.05,
            "ref_label": "Floor = 0.05",
            "ref_style": ":",
            "invert"   : False,
            "note"     : "Should decay from 1.0 to 0.05",
        },
    ]

    # ── Figure ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        f"DDQN Training Convergence  --  {n_ep} Episodes\n"
        f"(config: {os.path.basename(csv_path)})",
        fontsize=14, fontweight="bold", y=0.98,
    )
    axes_flat = axes.flatten()

    for ax, cfg in zip(axes_flat, panels):
        col = cfg["col"]
        if col not in df.columns:
            ax.set_title(f"{cfg['title']}  [NO DATA]")
            ax.text(0.5, 0.5, "Column not found in CSV",
                    ha="center", va="center", transform=ax.transAxes, color="red")
            continue

        raw    = df[col].values.astype(float)
        smooth = _rolling(df[col], window).values

        # Raw line (light)
        ax.plot(episodes, raw, color=cfg["color"], alpha=0.20,
                linewidth=0.8, label="Raw")

        # Smoothed line (bold)
        ax.plot(episodes, smooth, color=cfg["color"], alpha=0.95,
                linewidth=2.0, label=f"Rolling mean (w={window})")

        # Reference line
        if cfg.get("ref_val") is not None:
            ax.axhline(
                cfg["ref_val"],
                color="black", linestyle=cfg["ref_style"],
                linewidth=1.2, label=cfg["ref_label"], alpha=0.7,
            )

        # Log scale for loss
        if cfg.get("log"):
            nonzero = raw[raw > 0]
            if len(nonzero) > 0:
                ax.set_yscale("log")

        # Percent formatting for hit_rate
        if cfg.get("pct"):
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

        # Trend arrow annotation on last value
        last = float(smooth[-1])
        first_sm = float(smooth[0])
        slope = _trend_slope(df[col])
        arrow = "v" if slope < 0 else ("^" if slope > 0 else "->")
        colour_ok = cfg["invert"]     # green if falling for "invert" metrics
        correct_dir = (slope < 0) if cfg.get("invert") else (slope > 0 or col == "epsilon")
        ann_color = "#2ca02c" if correct_dir else "#d62728"

        ax.annotate(
            f"  last={last:.4f}  {arrow}",
            xy=(episodes[-1], last),
            xytext=(episodes[-1] * 0.80, last),
            fontsize=8, color=ann_color, fontweight="bold",
        )

        ax.set_title(cfg["title"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel(cfg["ylabel"], fontsize=9)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.tick_params(labelsize=8)

        # Subtitle note
        ax.text(0.02, 0.97, cfg["note"], transform=ax.transAxes,
                fontsize=7, color="grey", va="top", style="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved dashboard -> {out_path}")

    # ── Convergence report ───────────────────────────────────────────
    checks = _check_convergence(df)
    _SEP = "=" * 60
    _DIV = "-" * 60
    print()
    print(_SEP)
    print("  CONVERGENCE CHECKS")
    print(_DIV)
    all_passed = True
    for desc, passed, detail in checks:
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {desc}")
        print(f"       {detail}")
        if not passed:
            all_passed = False
    print(_DIV)
    if all_passed:
        print("  ALL CHECKS PASSED -- agent converged correctly")
    else:
        failed = sum(1 for _, p, _ in checks if not p)
        print(f"  {failed} check(s) FAILED -- review panels above")
    print(_SEP)

    # ── Per-phase summary ────────────────────────────────────────────
    q1 = n_ep // 4
    q3 = 3 * n_ep // 4

    print()
    print("  PHASE SUMMARY  (Early / Mid / Late episodes)")
    print(_DIV)
    print(f"  {'Metric':<20} {'Early (ep 1-'+str(q1)+')':>16}  "
          f"{'Mid (ep '+str(q1)+'-'+str(q3)+')':>18}  "
          f"{'Late (ep '+str(q3)+'-'+str(n_ep)+')':>18}")
    print(_DIV)

    for col, label, fmt in [
        ("reward",     "Reward",       "{:>+.2f}"),
        ("hit_rate",   "Hit Rate",     "{:>.1%}"),
        ("forwarded",  "Forwarded",    "{:>.1f}"),
        ("violations", "Violations",   "{:>.1f}"),
        ("avg_loss",   "Avg Loss",     "{:>.6f}"),
    ]:
        if col not in df.columns:
            continue
        s = df[col].values.astype(float)
        early = s[:q1].mean()
        mid   = s[q1:q3].mean()
        late  = s[q3:].mean()
        print(f"  {label:<20} {fmt.format(early):>16}  "
              f"{fmt.format(mid):>18}  "
              f"{fmt.format(late):>18}")

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
