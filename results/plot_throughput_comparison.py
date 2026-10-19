# -*- coding: utf-8 -*-
"""
plot_throughput_comparison.py
-----------------------------
Reads three ddqn_throughput_metrics CSV files (5000, 7000, 10000 request
checkpoints) and produces a 4-panel comparison dashboard:

  Panel 1 -- Hit Rate Stability (hit_rate_std%)
  Panel 2 -- Local Execution Rate (local_rate%)
  Panel 3 -- Cloud Escalation Rate (cloud_rate%)
  Panel 4 -- Total Revenue

Each panel overlays all three checkpoint runs as separate lines so
performance at each request load level can be compared directly.

Usage
-----
  python plot_throughput_comparison.py
  python plot_throughput_comparison.py --out results/throughput_comparison.png
  python plot_throughput_comparison.py
      --f5000  results/ddqn_throughput_metrics_5000.csv
      --f7000  results/ddqn_throughput_metrics_7000.csv
      --f10000 results/ddqn_throughput_metrics_10000.csv
      --out    results/throughput_comparison.png
"""

import argparse
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_ROOT = os.path.dirname(os.path.abspath(__file__))

_DEFAULT_5000  = os.path.join(_ROOT, "results", "ddqn_throughput_metrics_5000.csv")
_DEFAULT_7000  = os.path.join(_ROOT, "results", "ddqn_throughput_metrics_7000.csv")
_DEFAULT_10000 = os.path.join(_ROOT, "results", "ddqn_throughput_metrics_50k.csv")
_DEFAULT_OUT   = os.path.join(_ROOT, "results", "throughput_comparison-2.png")

# Colours and markers per checkpoint
_STYLES = {
    "Checkpoint 5000":  {"color": "#1f77b4", "marker": "o", "ls": "-"},
    "Checkpoint 7000":  {"color": "#ff7f0e", "marker": "s", "ls": "--"},
    "Checkpoint 10000": {"color": "#2ca02c", "marker": "^", "ls": "-."},
}

_SEP = "=" * 72
_DIV = "-" * 72


# ---------------------------------------------------------------------------
# Panel definitions
# ---------------------------------------------------------------------------

_PANELS = [
    {
        "col"          : "hit_rate_std",
        "title"        : "Hit Rate Stability\n(Std-Dev of Rolling Hit Rate)",
        "ylabel"       : "Hit Rate Std-Dev",
        "pct"          : True,
        "higher_better": False,
        "note"         : "Lower = more consistent cache policy",
    },
    {
        "col"          : "local_rate",
        "title"        : "Local Execution Rate\n(Requests served at origin leaf)",
        "ylabel"       : "Local Rate",
        "pct"          : True,
        "higher_better": True,
        "note"         : "Higher = DDQN caching at correct leaf nodes",
    },
    {
        "col"          : "cloud_rate",
        "title"        : "Cloud Escalation Rate\n(Requests escalated to cloud)",
        "ylabel"       : "Cloud Rate",
        "pct"          : True,
        "higher_better": False,
        "note"         : "Lower = fewer cache misses reaching cloud",
    },
    {
        "col"          : "total_revenue",
        "title"        : "Total Revenue\n(sum of revenue x omega executed)",
        "ylabel"       : "Revenue",
        "pct"          : False,
        "higher_better": True,
        "note"         : "Higher = more high-value requests served",
    },
]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_comparison(
    path_5000  : str = _DEFAULT_5000,
    path_7000  : str = _DEFAULT_7000,
    path_10000 : str = _DEFAULT_10000,
    out_path   : str = _DEFAULT_OUT,
) -> None:

    # -- Load CSVs --------------------------------------------------------
    files = {
        "Checkpoint 5000":  path_5000,
        "Checkpoint 7000":  path_7000,
        "Checkpoint 10000": path_10000,
    }

    data = {}
    for label, path in files.items():
        if not os.path.exists(path):
            print(f"  ERROR: file not found: {path}")
            sys.exit(1)
        data[label] = pd.read_csv(path)
        print(f"  Loaded {len(data[label])} rows from {path}")

    # -- Figure -----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "DDQN Throughput Comparison\n"
        "Checkpoint 5000 vs 7000 vs 10000 Requests",
        fontsize=14, fontweight="bold", y=0.99,
    )
    axes_flat = axes.flatten()

    for ax, cfg in zip(axes_flat, _PANELS):
        col = cfg["col"]

        for label, df in data.items():
            if col not in df.columns:
                continue
            sty = _STYLES[label]
            x   = df["n_requests"].values
            y   = df[col].values

            ax.plot(
                x, y,
                color=sty["color"],
                marker=sty["marker"],
                linestyle=sty["ls"],
                linewidth=2.0,
                markersize=7,
                label=label,
                zorder=3,
            )

            # Annotate last point
            last_x, last_y = float(x[-1]), float(y[-1])
            if cfg["pct"]:
                ann = f"{last_y*100:.1f}%"
            else:
                ann = f"{last_y:,.0f}"
            ax.annotate(
                ann,
                xy=(last_x, last_y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.5,
                color=sty["color"],
                fontweight="bold",
            )

        # Percent y-axis
        if cfg["pct"]:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

        # Direction indicator in subtitle
        direction = "(lower = better)" if not cfg["higher_better"] else "(higher = better)"

        ax.set_title(cfg["title"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Number of Requests", fontsize=9)
        ax.set_ylabel(cfg["ylabel"], fontsize=9)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.tick_params(labelsize=8)

        ax.text(
            0.02, 0.03,
            f"{direction}  |  {cfg['note']}",
            transform=ax.transAxes,
            fontsize=7, color="grey", va="bottom", style="italic",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Dashboard saved -> {out_path}")

    # -- Console comparison table -----------------------------------------
    print()
    print(_SEP)
    print("  COMPARISON TABLE  (values at each checkpoint's maximum load level)")
    print(_DIV)

    for label, df in data.items():
        last = df.iloc[-1]
        n    = int(last["n_requests"])
        print(f"\n  {label}  (N={n})")
        print(f"    Hit rate stability (std)  : {last['hit_rate_std']*100:>6.2f}%")
        print(f"    Local execution rate      : {last['local_rate']*100:>6.2f}%")
        print(f"    Cloud escalation rate     : {last['cloud_rate']*100:>6.2f}%")
        print(f"    Drop rate                 : {last['drop_rate']*100:>6.2f}%")
        print(f"    Total revenue             : {last['total_revenue']:>10,.2f}")
        print(f"    Revenue per request       : {last['rev_per_request']:>10.4f}")
        print(f"    mission_critical rev      : {last['rev_mission_critical']:>10,.2f}")
        print(f"    high rev                  : {last['rev_high']:>10,.2f}")
        print(f"    standard rev              : {last['rev_standard']:>10,.2f}")
        print(f"    best_effort rev           : {last['rev_best_effort']:>10,.2f}")

    print()
    print(_SEP)

    # -- Cross-checkpoint delta at shared load levels ---------------------
    print()
    print("  DELTA ANALYSIS  (Checkpoint 5000 as baseline)")
    print(_DIV)
    print(f"  {'N':>6}  {'Metric':<26}  "
          f"{'ckpt_5000':>10}  {'ckpt_7000':>10}  {'ckpt_10000':>11}  "
          f"{'7k delta':>9}  {'10k delta':>10}")
    print(_DIV)

    df5 = data["Checkpoint 5000"]
    df7 = data["Checkpoint 7000"]
    df10= data["Checkpoint 10000"]

    shared_n = sorted(
        set(df5["n_requests"]) & set(df7["n_requests"]) & set(df10["n_requests"])
    )

    for n in shared_n:
        r5  = df5[df5["n_requests"] == n].iloc[0]
        r7  = df7[df7["n_requests"] == n].iloc[0]
        r10 = df10[df10["n_requests"] == n].iloc[0]

        for col, label, pct in [
            ("local_rate",  "Local rate",  True),
            ("cloud_rate",  "Cloud rate",  True),
            ("total_revenue","Revenue",    False),
        ]:
            v5,  v7,  v10 = float(r5[col]), float(r7[col]), float(r10[col])
            d7  = v7  - v5
            d10 = v10 - v5
            if pct:
                print(f"  {n:>6}  {label:<26}  "
                      f"{v5*100:>9.2f}%  {v7*100:>9.2f}%  {v10*100:>10.2f}%  "
                      f"{d7*100:>+8.2f}pp  {d10*100:>+9.2f}pp")
            else:
                print(f"  {n:>6}  {label:<26}  "
                      f"{v5:>10,.0f}  {v7:>10,.0f}  {v10:>11,.0f}  "
                      f"{d7:>+9,.0f}  {d10:>+10,.0f}")
        print()

    print(_SEP)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="plot_throughput_comparison.py",
        description=(
            "Plot DDQN throughput comparison across three checkpoint CSV files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python plot_throughput_comparison.py\n\n"
            "  python plot_throughput_comparison.py \\\n"
            "      --f5000  results/ddqn_throughput_metrics_5000.csv \\\n"
            "      --f7000  results/ddqn_throughput_metrics_7000.csv \\\n"
            "      --f10000 results/ddqn_throughput_metrics_10000.csv \\\n"
            "      --out    results/throughput_comparison.png\n"
        ),
    )
    p.add_argument("--f5000",  type=str, default=_DEFAULT_5000,
                   help=f"5000-request checkpoint CSV (default: {_DEFAULT_5000})")
    p.add_argument("--f7000",  type=str, default=_DEFAULT_7000,
                   help=f"7000-request checkpoint CSV (default: {_DEFAULT_7000})")
    p.add_argument("--f10000", type=str, default=_DEFAULT_10000,
                   help=f"10000-request checkpoint CSV (default: {_DEFAULT_10000})")
    p.add_argument("--out",    type=str, default=_DEFAULT_OUT,
                   help=f"Output PNG path (default: {_DEFAULT_OUT})")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(_SEP)
    print("  DDQN Throughput Comparison Plotter")
    print(_SEP)
    plot_comparison(
        path_5000  = args.f5000,
        path_7000  = args.f7000,
        path_10000 = args.f10000,
        out_path   = args.out,
    )
