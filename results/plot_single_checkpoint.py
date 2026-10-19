# -*- coding: utf-8 -*-
"""
plot_single_checkpoint.py
--------------------------
Reads one ddqn_throughput_metrics CSV file and produces a 4-panel
dashboard showing performance at selected load levels.

Panels
------
  1. Hit Rate Mean      -- fraction of requests executed  (higher = better)
  2. Local Execution    -- fraction served at origin leaf  (higher = better)
  3. Cloud Escalation   -- fraction escalated to cloud     (lower  = better)
  4. Total Revenue      -- sum(revenue x omega) executed   (higher = better)

Each panel shows:
  - A faint line across ALL available load levels in the CSV
  - Bold line + hollow circle markers at the requested focus points
  - Value annotations with white box on every focus point
  - Vertical dashed guide lines at each focus point

Usage
-----
  # Default: ddqn_throughput_metrics_10000.csv, focus 100,1000,5000,7000,10000
  python plot_single_checkpoint.py

  # Custom CSV and focus points
  python plot_single_checkpoint.py
      --csv   results/ddqn_throughput_metrics_10000.csv
      --focus 100 1000 5000 7000 10000
      --out   results/single_checkpoint_dashboard.png
      --title "DDQN Checkpoint -- 10000 Requests"
"""

import argparse
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_ROOT        = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CSV = os.path.join(_ROOT, "results", "ddqn_throughput_metrics_50k.csv")
_DEFAULT_OUT = os.path.join(_ROOT, "results", "single_checkpoint_dashboard_3.png")
_DEFAULT_FOCUS = [100, 1000, 5000, 7000, 10000]

_SEP = "=" * 72
_DIV = "-" * 72

# ---------------------------------------------------------------------------
# Panel definitions
# ---------------------------------------------------------------------------

_PANELS = [
    {
        "col"          : "hit_rate_mean",
        "title"        : "Hit Rate Mean\n(Executed / Total Requests)",
        "ylabel"       : "Hit Rate",
        "color"        : "#1f77b4",
        "pct"          : True,
        "higher_better": True,
        "ref_val"      : 0.90,
        "ref_label"    : "90% target",
        "note"         : "Higher = more requests served without cloud fallback",
    },
    {
        "col"          : "local_rate",
        "title"        : "Local Execution Rate\n(Served at Origin Leaf Node)",
        "ylabel"       : "Local Rate",
        "color"        : "#2ca02c",
        "pct"          : True,
        "higher_better": True,
        "ref_val"      : None,
        "ref_label"    : None,
        "note"         : "Higher = DDQN cached the right service at the right leaf",
    },
    {
        "col"          : "cloud_rate",
        "title"        : "Cloud Escalation Rate\n(Forwarded All the Way to Cloud)",
        "ylabel"       : "Cloud Rate",
        "color"        : "#d62728",
        "pct"          : True,
        "higher_better": False,
        "ref_val"      : None,
        "ref_label"    : None,
        "note"         : "Lower = fewer cache misses reaching cloud",
    },
    {
        "col"          : "total_revenue",
        "title"        : "Total Revenue\n(sum of revenue x omega for executed requests)",
        "ylabel"       : "Revenue",
        "color"        : "#ff7f0e",
        "pct"          : False,
        "higher_better": True,
        "ref_val"      : None,
        "ref_label"    : None,
        "note"         : "Higher = more high-value requests served",
    },
]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_single_checkpoint(
    csv_path  : str  = _DEFAULT_CSV,
    focus_pts : list = _DEFAULT_FOCUS,
    out_path  : str  = _DEFAULT_OUT,
    title     : str  = "",
) -> None:

    if not os.path.exists(csv_path):
        print(f"  ERROR: file not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows from {csv_path}")

    # Clamp focus points to rows that actually exist in the CSV
    available_n = set(df["n_requests"].tolist())
    focus_pts   = sorted(set(focus_pts) & available_n)
    focus_df    = df[df["n_requests"].isin(focus_pts)]

    if not focus_pts:
        print(f"  WARNING: none of the requested focus points found in CSV.")
        print(f"  Available load levels: {sorted(available_n)}")
        focus_df = df
    else:
        print(f"  Focus points used: {focus_pts}")

    # -- Figure -----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    auto_title = title if title else (
        f"DDQN Single Checkpoint Analysis\n"
        f"{os.path.basename(csv_path)}  |  "
        f"Focus: N = {', '.join(str(n) for n in focus_pts)}"
    )
    fig.suptitle(auto_title, fontsize=13, fontweight="bold", y=0.99)
    axes_flat = axes.flatten()

    x_all = df["n_requests"].values

    for ax, cfg in zip(axes_flat, _PANELS):
        col   = cfg["col"]
        color = cfg["color"]

        if col not in df.columns:
            ax.set_title(f"{cfg['title']}  [NO DATA]")
            ax.text(0.5, 0.5, "Column not found",
                    ha="center", va="center", transform=ax.transAxes, color="red")
            continue

        y_all = df[col].values

        # Vertical guide lines at focus points (drawn first, behind everything)
        for xv in focus_pts:
            ax.axvline(xv, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)

        # Full faint trend line across all load levels
        ax.plot(
            x_all, y_all,
            color=color, linewidth=1.6, linestyle="-",
            alpha=0.35, zorder=2, label="All load levels",
        )

        # Bold line + hollow circle markers at focus points
        if len(focus_df) > 0:
            x_focus = focus_df["n_requests"].values
            y_focus = focus_df[col].values

            ax.plot(
                x_focus, y_focus,
                color=color, linewidth=2.4, linestyle="-",
                marker="o", markersize=9,
                markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=2.5,
                zorder=4, label="Focus points",
            )

            # Value annotations on each focus point
            for xv, yv in zip(x_focus, y_focus):
                ann    = f"{yv*100:.1f}%" if cfg["pct"] else f"{yv:,.0f}"
                va     = "bottom" if cfg["higher_better"] else "top"
                offset = (0, 9)  if cfg["higher_better"] else (0, -9)

                ax.annotate(
                    ann,
                    xy=(xv, yv),
                    xytext=offset,
                    textcoords="offset points",
                    ha="center", va=va,
                    fontsize=8.5, fontweight="bold",
                    color=color,
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor="white",
                        edgecolor=color,
                        alpha=0.88,
                        linewidth=0.9,
                    ),
                )

        # Optional reference line (e.g. 90% hit rate target)
        if cfg.get("ref_val") is not None:
            ax.axhline(
                cfg["ref_val"],
                color="black", linestyle=":", linewidth=1.3,
                alpha=0.6, label=cfg["ref_label"],
            )

        # Percent y-axis formatting
        if cfg["pct"]:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

        direction = "(higher = better)" if cfg["higher_better"] else "(lower = better)"

        ax.set_title(cfg["title"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Number of Requests", fontsize=9)
        ax.set_ylabel(cfg["ylabel"], fontsize=9)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.25, linestyle=":")
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
    print(f"  Dashboard saved -> {out_path}")

    # -- Console table at focus points ------------------------------------
    print()
    print(_SEP)
    print(f"  FOCUS POINT SUMMARY  --  {os.path.basename(csv_path)}")
    print(_DIV)
    print(f"  {'N':>7}  {'HitRate%':>9}  {'Local%':>8}  {'Cloud%':>8}  "
          f"{'Drop%':>7}  {'Revenue':>12}  {'Rev/Req':>9}")
    print(_DIV)

    for _, row in focus_df.iterrows():
        print(
            f"  {int(row['n_requests']):>7}  "
            f"{row['hit_rate_mean']*100:>8.2f}%  "
            f"{row['local_rate']*100:>7.2f}%  "
            f"{row['cloud_rate']*100:>7.2f}%  "
            f"{row['drop_rate']*100:>6.2f}%  "
            f"{row['total_revenue']:>12,.2f}  "
            f"{row['rev_per_request']:>9.4f}"
        )

    print(_DIV)

    # -- Delta from first to last focus point -----------------------------
    if len(focus_df) >= 2:
        first = focus_df.iloc[0]
        last  = focus_df.iloc[-1]
        print(f"\n  DELTA  (N={int(first['n_requests'])} -> N={int(last['n_requests'])})")
        print(_DIV)
        for col, label, pct in [
            ("hit_rate_mean",  "Hit rate mean",         True),
            ("local_rate",     "Local execution rate",  True),
            ("cloud_rate",     "Cloud escalation rate", True),
            ("drop_rate",      "Drop rate",             True),
            ("total_revenue",  "Total revenue",         False),
            ("rev_per_request","Revenue per request",   False),
        ]:
            fv = float(first[col])
            lv = float(last[col])
            d  = lv - fv
            if pct:
                print(f"  {label:<28}  {fv*100:>6.2f}%  ->  {lv*100:>6.2f}%  "
                      f"({d*100:>+6.2f} pp)")
            else:
                print(f"  {label:<28}  {fv:>10,.2f}  ->  {lv:>10,.2f}  "
                      f"({d:>+10,.2f})")

    print(_SEP)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="plot_single_checkpoint.py",
        description=(
            "Plot DDQN throughput metrics from a single CSV file "
            "at selected load levels."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python plot_single_checkpoint.py\n\n"
            "  python plot_single_checkpoint.py \\\n"
            "      --csv   results/ddqn_throughput_metrics_10000.csv \\\n"
            "      --focus 100 1000 5000 7000 10000 \\\n"
            "      --out   results/single_checkpoint_dashboard.png\n"
        ),
    )
    p.add_argument(
        "--csv", type=str, default=_DEFAULT_CSV,
        help=f"Path to throughput metrics CSV (default: {_DEFAULT_CSV})",
    )
    p.add_argument(
        "--focus", type=int, nargs="+", default=_DEFAULT_FOCUS,
        help="Load levels to highlight (default: 100 1000 5000 7000 10000)",
    )
    p.add_argument(
        "--out", type=str, default=_DEFAULT_OUT,
        help=f"Output PNG path (default: {_DEFAULT_OUT})",
    )
    p.add_argument(
        "--title", type=str, default="",
        help="Custom plot title (default: auto-generated from filename)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(_SEP)
    print("  DDQN Single Checkpoint Dashboard")
    print(_SEP)
    plot_single_checkpoint(
        csv_path  = args.csv,
        focus_pts = sorted(set(args.focus)),
        out_path  = args.out,
        title     = args.title,
    )
