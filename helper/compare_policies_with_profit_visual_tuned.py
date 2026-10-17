"""
compare_policies_with_profit.py
--------------------------------
Combined policy comparison with profit analysis.

Generates:
1. Trend charts: Revenue, Drop Rate, Cloud Escalation, Local Hit Rate
2. Bar charts: Profit, Revenue, Placement Cost, Forward Cost, SLA Violation
3. CSV files with complete metrics

All plots saved at 300 DPI with consistent dimensions.

Usage:
  python compare_policies_with_profit.py
  python compare_policies_with_profit.py --steps 100 1000 5000 10000
"""

import argparse
import csv
import os
import sys
import random
from collections import defaultdict
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IEEE standard plot settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['axes.linewidth'] = 1.8
plt.rcParams['grid.linewidth'] = 1.0
plt.rcParams['lines.linewidth'] = 4.8

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from edge_caching.config import (
    NUM_LEVELS, NUM_CHILDREN, NODE_CACHE_CAP,
    PROVISIONING_INTERVAL, RANDOM_SEED, RESULTS_DIR,
    LAMBDA_PLACEMENT, LAMBDA_FORWARD, LAMBDA_SLA,
)
from edge_caching.core.node import EdgeNode, ACTION_INSERT, ACTION_DELETE, ACTION_KEEP
from edge_caching.core.service import Service
from edge_caching.core.request import Request
from edge_caching.network.topology import NetworkTopology
from edge_caching.control.orchestrator import EdgeOrchestrator
from edge_caching.agents.ddqn_agent import DDQNAgent
from edge_caching.utils.simulation.request_generator import load_requests_from_csv
from edge_caching.policies.lca_placement import make_lca_policy
from edge_caching.policies.top_down_placement import make_top_down_policy

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_CHECKPOINT = os.path.join(_ROOT, "checkpoints", "ddqn_finetuned_resumed_500epv3.pt")
_DEFAULT_REQUESTS   = os.path.join(_ROOT, "training", "data", "sample_requests_10000.csv")
_DEFAULT_SERVICES   = os.path.join(_ROOT, "training", "data", "sample_services.csv")
_DEFAULT_OUT_DIR    = os.path.join(_ROOT, RESULTS_DIR)
_DEFAULT_STEPS      = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

_POLICIES = ["DDQN", "LCA", "TopDown", "Greedy"]
_TIERS    = ["mission_critical", "high", "standard", "best_effort"]

_SEP = "=" * 80
_DIV = "-" * 80

# Chart dimensions (consistent across all plots)
CHART_WIDTH = 15
CHART_HEIGHT = 10
CHART_DPI = 300


# ---------------------------------------------------------------------------
# Service loader
# ---------------------------------------------------------------------------

def _load_services(csv_path: str) -> Dict[int, Service]:
    """Load service catalogue from CSV."""
    catalogue = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = Service(
                service_id      = int(row["service_id"]),
                size            = float(row.get("size_mb", row.get("size", 1000))),
                popularity      = float(row["popularity"]),
                criticality     = row["criticality"].strip(),
                revenue         = float(row["revenue"]),
                init_cpu_cycles = float(row.get("init_cpu_cycles", 50000)),
            )
            catalogue[s.service_id] = s
    return catalogue


# ---------------------------------------------------------------------------
# World reset
# ---------------------------------------------------------------------------

def _reset_world(topology: NetworkTopology, catalogue: Dict[int, Service]) -> None:
    """Reset topology for fresh evaluation."""
    for node in [topology.cloud_root] + topology.all_nodes:
        node.cache.clear()
        node.cache_sizes.clear()
        node._queue.clear()
        node.total_revenue = 0.0
        node.reset_interval_stats()
    for svc in catalogue.values():
        svc.reset_interval_counters()
    topology.prime_cloud_cache(catalogue)
    topology.priority_init(catalogue)


# ---------------------------------------------------------------------------
# Tier classifier
# ---------------------------------------------------------------------------

def _classify_tier(req: Request) -> str:
    """Classify where request was executed."""
    if req.status == "dropped":
        return "dropped"
    if req.executed_at_node == req.origin_node_id:
        return "leaf"
    if req.executed_at_node == "CLOUD":
        return "cloud"
    # Check node log for escalation patterns
    for _, evt in req.node_log:
        if evt == "cache_miss_peer":
            return "peer"
        if evt in ("cache_miss_parent", "cache_miss_cloud"):
            return "parent"
    return "parent"


# ---------------------------------------------------------------------------
# Policy functions
# ---------------------------------------------------------------------------

def make_ddqn_policy(agent: DDQNAgent):
    """Create DDQN policy function."""
    def _policy(node: EdgeNode, service: Service, snap: dict) -> int:
        state = DDQNAgent.build_state_vector(service, snap)
        return agent.select_action(state)
    return _policy


def greedy_policy(node: EdgeNode, service: Service, snap: dict) -> int:
    """Greedy demand-reactive policy."""
    svc_id = service.service_id
    fwd = snap["forwarded_count"].get(svc_id, 0)
    exe = snap["executed_count"].get(svc_id, 0)
    srv = snap["served_count"].get(svc_id, 0)
    is_cached = svc_id in snap["cached_services"]

    if not is_cached and fwd > 0:
        return ACTION_INSERT
    if is_cached and exe == 0 and srv == 0 and fwd == 0:
        return ACTION_DELETE
    return ACTION_KEEP


# ---------------------------------------------------------------------------
# Heuristic policy runner with cost tracking
# ---------------------------------------------------------------------------

def _run_heuristic_policy(
    requests      : List[Request],
    topology      : NetworkTopology,
    catalogue     : Dict[int, Service],
    policy_fn,
    prov_interval : int,
    policy_name   : str,
) -> dict:
    """Run heuristic policy and track all metrics."""
    _reset_world(topology, catalogue)

    tiers = defaultdict(int)
    revenue_total = 0.0
    placement_cost = 0.0
    forwarding_cost = 0.0
    violation_cost = 0.0
    insert_count = 0
    delete_count = 0
    keep_count = 0

    for idx, req in enumerate(requests):
        # Track requests for policy-specific logic
        if hasattr(policy_fn, 'lca_instance'):
            policy_fn.lca_instance.track_request(req.service_id, req.origin_node_id)
        if hasattr(policy_fn, 'td_instance'):
            policy_fn.td_instance.track_request(req.service_id, req.origin_node_id)

        # Fast process
        topology.node_map[req.origin_node_id].receive_request(req, catalogue)

        tier = _classify_tier(req)
        tiers[tier] += 1

        # Revenue
        if req.status == "executed":
            svc = catalogue[req.service_id]
            rv = svc.revenue * svc.omega
            revenue_total += rv

        # Forwarding costs
        if tier in ["peer", "parent", "cloud"]:
            forwarding_cost += LAMBDA_FORWARD

        # Violation costs
        if tier == "dropped":
            violation_cost += LAMBDA_SLA

        # Slow process
        if (idx + 1) % prov_interval == 0:
            if hasattr(policy_fn, 'td_instance'):
                policy_fn.td_instance.compute_all_decisions(catalogue)

            for node in topology.all_nodes:
                snap = node.get_stats_snapshot()
                for svc_id, service in catalogue.items():
                    action = policy_fn(node, service, snap)

                    if action == ACTION_INSERT:
                        insert_count += 1
                        placement_cost += LAMBDA_PLACEMENT
                    elif action == ACTION_DELETE:
                        delete_count += 1
                        placement_cost += LAMBDA_PLACEMENT
                    else:
                        keep_count += 1

                    node.apply_action(action, service)

            # Update popularity
            total_demand = max(sum(s.demand_count for s in catalogue.values()), 1)
            for service in catalogue.values():
                service.update_popularity(total_demand)
                service.reset_interval_counters()
            for node in topology.all_nodes:
                node.reset_interval_stats()

            # Reset policy tracking
            if hasattr(policy_fn, 'lca_instance'):
                policy_fn.lca_instance.reset_interval()
            if hasattr(policy_fn, 'td_instance'):
                policy_fn.td_instance.reset_interval()

    total = len(requests)
    executed = total - tiers["dropped"]
    total_cost = placement_cost + forwarding_cost + violation_cost
    # For policy comparison: profit = revenue (no business costs deducted)
    # Costs are tracked for analysis but don't reduce profit
    profit = revenue_total

    return {
        "policy": policy_name,
        "n_requests": total,
        "executed": executed,
        "dropped": tiers["dropped"],
        "local_rate": round(tiers["leaf"] / max(total, 1), 6),
        "cloud_rate": round(tiers["cloud"] / max(total, 1), 6),
        "drop_rate": round(tiers["dropped"] / max(total, 1), 6),
        "revenue": round(revenue_total, 2),
        "placement_cost": round(placement_cost, 2),
        "forwarding_cost": round(forwarding_cost, 2),
        "violation_cost": round(violation_cost, 2),
        "total_cost": round(total_cost, 2),
        "profit": round(profit, 2),  # Profit = Revenue for policy comparison
    }


# ---------------------------------------------------------------------------
# DDQN policy runner
# ---------------------------------------------------------------------------

def _run_ddqn_policy(
    requests      : List[Request],
    topology      : NetworkTopology,
    catalogue     : Dict[int, Service],
    orchestrator  : EdgeOrchestrator,
    prov_interval : int,
) -> dict:
    """Run DDQN policy with cost tracking."""
    _reset_world(topology, catalogue)

    orchestrator._prev_snapshots = {}
    orchestrator._last_provision_sim_time = 0.0
    orchestrator.agent.epsilon = 0.0

    tiers = defaultdict(int)
    revenue_total = 0.0
    placement_cost = 0.0
    forwarding_cost = 0.0
    violation_cost = 0.0
    insert_count = 0
    delete_count = 0

    total = len(requests)

    for idx, req in enumerate(requests):
        topology.node_map[req.origin_node_id].receive_request(req, catalogue)

        tier = _classify_tier(req)
        tiers[tier] += 1

        if req.status == "executed":
            svc = catalogue[req.service_id]
            rv = svc.revenue * svc.omega
            revenue_total += rv

        if tier in ["peer", "parent", "cloud"]:
            forwarding_cost += LAMBDA_FORWARD

        if tier == "dropped":
            violation_cost += LAMBDA_SLA

        if (idx + 1) % prov_interval == 0:
            # Track actions
            for node in orchestrator.nodes:
                snap = node.get_stats_snapshot()
                for svc_id, service in catalogue.items():
                    state = DDQNAgent.build_state_vector(service, snap)
                    action = orchestrator.agent.select_action(state)

                    if action == ACTION_INSERT:
                        insert_count += 1
                        placement_cost += LAMBDA_PLACEMENT
                    elif action == ACTION_DELETE:
                        placement_cost += LAMBDA_PLACEMENT

                    node.apply_action(action, service)

            total_demand = max(sum(s.demand_count for s in catalogue.values()), 1)
            for service in catalogue.values():
                service.update_popularity(total_demand)
                service.reset_interval_counters()
            for node in topology.all_nodes:
                node.reset_interval_stats()

    executed = total - tiers["dropped"]
    total_cost = placement_cost + forwarding_cost + violation_cost
    # For policy comparison: profit = revenue (no business costs deducted)
    # Costs are tracked for analysis but don't reduce profit
    profit = revenue_total - total_cost

    return {
        "policy": "DDQN",
        "n_requests": total,
        "executed": executed,
        "dropped": tiers["dropped"],
        "local_rate": round(tiers["leaf"] / max(total, 1), 6),
        "cloud_rate": round(tiers["cloud"] / max(total, 1), 6),
        "drop_rate": round(tiers["dropped"] / max(total, 1), 6),
        "revenue": round(revenue_total, 2),
        "placement_cost": round(placement_cost, 2),
        "forwarding_cost": round(forwarding_cost, 2),
        "violation_cost": round(violation_cost, 2),
        "total_cost": round(total_cost, 2),
        "profit": round(profit, 2),  # Profit = Revenue for policy comparison
    }


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_trend_charts(results: Dict[str, List[dict]], out_dir: str):
    """Generate trend charts for Revenue, Drop Rate, Cloud Rate, Local Rate."""

    metrics = [
        ("revenue", "Revenue", "Revenue Comparison Across Policies", False, True),
        ("drop_rate", "Drop Rate (%)", "Drop Rate Comparison Across Policies", True, False),
        ("cloud_rate", "Cloud Escalation Rate (%)", "Cloud Escalation Comparison Across Policies", True, False),
        ("local_rate", "Local Execution Rate (%)", "Local Execution Comparison Across Policies", True, False),
    ]

    colors = {"DDQN": "#1f77b4", "LCA": "#2ca02c", "TopDown": "#ff7f0e", "Greedy": "#d62728"}
    markers = {"DDQN": "o", "LCA": "D", "TopDown": "s", "Greedy": "^"}
    linestyles = {"DDQN": "-", "LCA": "--", "TopDown": "-.", "Greedy": ":"}

    for metric_key, ylabel, title, is_percentage, is_currency in metrics:
        fig, ax = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))

        for policy_name in _POLICIES:
            if policy_name not in results:
                continue

            rows = results[policy_name]
            xs = [r["n_requests"] for r in rows]
            ys = [r[metric_key] for r in rows]

            if is_percentage:
                ys = [y * 100 for y in ys]

            ax.plot(xs, ys,
                   color=colors.get(policy_name, "#000"),
                   marker=markers.get(policy_name, "o"),
                   linestyle=linestyles.get(policy_name, "-"),
                   linewidth=5.2, markersize=13, label=policy_name,
                   markeredgewidth=2.4, markeredgecolor='white',
                   markerfacecolor=colors.get(policy_name, "#000"),
                   solid_capstyle='round', zorder=3)

            # Light fill to make the trend visually fuller
            ax.fill_between(xs, ys, 0, color=colors.get(policy_name, "#000"), alpha=0.08, zorder=1)

            # Annotate final value at 10,000 requests
            if xs and xs[-1] == 10000:
                final_val = ys[-1]
                if is_percentage:
                    label_text = f'{final_val:.1f}%'
                else:
                    label_text = f'{final_val:,.0f}'

                ax.annotate(label_text,
                           xy=(xs[-1], ys[-1]),
                           xytext=(12, 0), textcoords='offset points',
                           fontsize=13, fontweight='bold',
                           color=colors.get(policy_name, "#000"),
                           bbox=dict(boxstyle='round,pad=0.4', 
                                   facecolor='white', 
                                   edgecolor=colors.get(policy_name, "#000"),
                                   alpha=0.95,
                                   linewidth=1.8))

        ax.set_xlabel("Number of Requests", fontsize=20, fontweight="bold", labelpad=10)
        ax.set_ylabel(ylabel, fontsize=20, fontweight="bold", labelpad=10)
        ax.set_title(title, fontsize=21, fontweight="bold", pad=22)
        ax.legend(fontsize=15, loc="best", framealpha=0.97, edgecolor='black', frameon=True)
        ax.grid(True, alpha=0.35, linestyle='--', linewidth=1.0)

        # Format axes
        ax.set_xlim(left=0, right=10500)
        ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
        ax.tick_params(axis='both', labelsize=18, width=1.8, length=7, pad=6)

        if is_currency:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

        # Ensure y-axis starts at 0
        ylim = ax.get_ylim()
        ax.set_ylim(bottom=0, top=ylim[1] * 1.1)

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"trend_{metric_key}.png")
        plt.savefig(out_path, dpi=CHART_DPI, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")


def plot_ddqn_cost_breakdown_bars(results: Dict[str, List[dict]], out_dir: str):
    """Generate grouped bar chart showing DDQN cost breakdown at multiple load levels."""

    if "DDQN" not in results or not results["DDQN"]:
        print("  No DDQN data for cost breakdown")
        return

    # Select key load points
    target_loads = [1000, 5000, 7000, 10000]
    ddqn_data = []

    for load in target_loads:
        matching = [r for r in results["DDQN"] if r["n_requests"] == load]
        if matching:
            ddqn_data.append(matching[0])

    if not ddqn_data:
        print("  No DDQN data at target loads")
        return

    fig, ax = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))

    # Prepare data
    x_labels = [f'{d["n_requests"]:,}' for d in ddqn_data]
    revenues = [d["revenue"] for d in ddqn_data]
    profits = [d["profit"] for d in ddqn_data]
    placement_costs = [d["placement_cost"] for d in ddqn_data]
    forwarding_costs = [d["forwarding_cost"] for d in ddqn_data]
    violation_costs = [d["violation_cost"] for d in ddqn_data]

    x_pos = np.arange(len(x_labels))
    width = 0.15

    # Create grouped bars
    bars1 = ax.bar(x_pos - 2*width, revenues, width, label='Revenue', color='#2ecc71', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_pos - width, profits, width, label='Profit', color='#3498db', edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x_pos, placement_costs, width, label='Placement Cost', color='#e74c3c', edgecolor='black', linewidth=1.5)
    bars4 = ax.bar(x_pos + width, forwarding_costs, width, label='Forwarding Cost', color='#f39c12', edgecolor='black', linewidth=1.5)
    bars5 = ax.bar(x_pos + 2*width, violation_costs, width, label='SLA Violation Cost', color='#9b59b6', edgecolor='black', linewidth=1.5)

    # Calculate offset for 4-pixel gap (as percentage of max value)
    all_values = revenues + profits + placement_costs + forwarding_costs + violation_costs
    max_value = max(all_values) if all_values else 1
    offset = max_value * 0.005  # Small offset for visual gap

    # Add value labels (vertical text with gap above bars)
    for bars in [bars1, bars2, bars3, bars4, bars5]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                   f'{height:,.0f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   rotation=90)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Requests", fontsize=15, fontweight="bold")
    ax.set_ylabel("Amount", fontsize=15, fontweight="bold")
    ax.set_title("DDQN Cost Breakdown Across Load Levels", fontsize=16, fontweight="bold", pad=20)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax.legend(fontsize=12, loc="upper left", framealpha=0.95, edgecolor='black', frameon=True)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.tick_params(axis='both', labelsize=15, width=1.5, length=6)

    # Add space for labels
    ylim = ax.get_ylim()
    ax.set_ylim(bottom=0, top=ylim[1] * 1.2)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "ddqn_cost_breakdown_all_loads.png")
    plt.savefig(out_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_ddqn_individual_cost_trends(results: Dict[str, List[dict]], out_dir: str):
    """Generate separate trend charts for each DDQN cost component."""

    if "DDQN" not in results or not results["DDQN"]:
        print("  No DDQN data for individual costs")
        return

    rows = results["DDQN"]
    xs = [r["n_requests"] for r in rows]

    cost_metrics = [
        ("placement_cost", "Placement Cost", "DDQN Placement Cost Trend", '#e74c3c'),
        ("forwarding_cost", "Forwarding Cost", "DDQN Forwarding Cost Trend", '#f39c12'),
        ("violation_cost", "SLA Violation Cost", "DDQN SLA Violation Cost Trend", '#9b59b6'),
    ]

    for metric_key, metric_name, title, color in cost_metrics:
        fig, ax = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))

        ys = [r[metric_key] for r in rows]

        # Plot trend line
        ax.plot(xs, ys, color=color, marker='o', linewidth=3.5, markersize=10,
               markeredgewidth=2, markeredgecolor='white', label=metric_name)

        # Fill area under curve
        ax.fill_between(xs, ys, alpha=0.2, color=color)

        # Annotate final value
        if xs and xs[-1] == 10000:
            final_val = ys[-1]
            ax.annotate(f'{final_val:,.0f}',
                       xy=(xs[-1], ys[-1]),
                       xytext=(12, 10), textcoords='offset points',
                       fontsize=12, fontweight='bold', color=color,
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                               edgecolor=color, alpha=0.95, linewidth=1.5))

        # Calculate rate of change
        if len(xs) > 1:
            rates = []
            for i in range(1, len(xs)):
                rate = (ys[i] - ys[i-1]) / (xs[i] - xs[i-1]) * 1000
                rates.append(rate)

            avg_rate = np.mean(rates)
            ax.text(0.02, 0.95, f'Avg Rate: {avg_rate:.2f}/1K req',
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.95))

        # Trend annotation
        if len(ys) > 1:
            trend = "Increasing" if ys[-1] > ys[0] else "Decreasing" if ys[-1] < ys[0] else "Stable"
            change_pct = ((ys[-1] - ys[0]) / max(ys[0], 1)) * 100
            ax.text(0.98, 0.05, f'Trend: {trend}\nChange: {change_pct:+.1f}%',
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   horizontalalignment='right', verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=1.2))

        ax.set_xlabel("Number of Requests", fontsize=20, fontweight="bold", labelpad=10)
        ax.set_ylabel(metric_name, fontsize=15, fontweight="bold")
        ax.set_title(title, fontsize=21, fontweight="bold", pad=22)
        ax.grid(True, alpha=0.35, linestyle='--', linewidth=1.0)
        ax.set_xlim(left=0, right=10500)
        ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        ax.tick_params(axis='both', labelsize=18, width=1.8, length=7, pad=6)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=15, loc="best", framealpha=0.97, edgecolor='black', frameon=True)

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"ddqn_{metric_key}_trend.png")
        plt.savefig(out_path, dpi=CHART_DPI, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")


def plot_profit_comparison_bar(results: Dict[str, List[dict]], out_dir: str):
    """Generate bar chart comparing profit across all policies at maximum load."""

    # Get last entry (max load) for each policy
    max_load_data = {}
    for policy_name in _POLICIES:
        if policy_name in results and results[policy_name]:
            max_load_data[policy_name] = results[policy_name][-1]

    if not max_load_data:
        print("  No data for profit comparison bars")
        return

    fig, ax = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))

    policies = list(max_load_data.keys())
    profits = [max_load_data[p]["profit"] for p in policies]
    colors_dict = {"DDQN": "#1f77b4", "LCA": "#2ca02c", "TopDown": "#ff7f0e", "Greedy": "#d62728"}
    bar_colors = [colors_dict.get(p, "#888") for p in policies]

    x_pos = np.arange(len(policies))
    bars = ax.bar(x_pos, profits, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=2)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, profits)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(profits) * 0.02,
               f'{val:,.0f}', ha='center', va='bottom', fontsize=13, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor='black', linewidth=1.2))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(policies, fontsize=14, fontweight="bold")
    ax.set_ylabel("Profit", fontsize=15, fontweight="bold")
    ax.set_title(f"Profit Comparison @ {max_load_data[policies[0]]['n_requests']:,} Requests", 
                fontsize=16, fontweight="bold", pad=20)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.tick_params(axis='both', labelsize=15, width=1.5, length=6)

    # Add space at top for labels
    ylim = ax.get_ylim()
    ax.set_ylim(bottom=0, top=ylim[1] * 1.12)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "profit_comparison.png")
    plt.savefig(out_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    checkpoint    : str,
    requests_csv  : str,
    services_csv  : str,
    steps         : List[int],
    prov_interval : int,
    seed          : int,
    out_dir       : str,
):
    random.seed(seed)
    np.random.seed(seed)

    print(_SEP)
    print("  Combined Policy Comparison with Profit Analysis")
    print(_DIV)
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Requests   : {requests_csv}")
    print(f"  Services   : {services_csv}")
    print(f"  Load steps : {steps}")
    print(f"  Output dir : {out_dir}")
    print(_SEP)

    # Load data
    catalogue = _load_services(services_csv)
    print(f"  Loaded {len(catalogue)} services")

    topology = NetworkTopology(NUM_LEVELS, NUM_CHILDREN, NODE_CACHE_CAP)
    topology.prime_cloud_cache(catalogue)
    topology.priority_init(catalogue)
    print(f"  Topology: {len(topology.all_nodes)} edge nodes, {len(topology.leaf_nodes)} leaves")

    all_requests = load_requests_from_csv(requests_csv, catalogue, topology.node_map)
    print(f"  Loaded {len(all_requests)} requests\n")

    steps = sorted(set(min(n, len(all_requests)) for n in steps))

    # Build orchestrator
    orchestrator = EdgeOrchestrator(nodes=topology.all_nodes, catalogue=catalogue)
    if os.path.exists(checkpoint):
        orchestrator.load_checkpoint(checkpoint)
        orchestrator.agent.epsilon = 0.0
        print(f"  DDQN loaded from {checkpoint}")
    else:
        print(f"  WARNING: Checkpoint not found")

    # Create policies
    ddqn_fn = make_ddqn_policy(orchestrator.agent)
    lca_fn = make_lca_policy(topology)
    td_fn = make_top_down_policy(topology)
    greedy_fn = greedy_policy

    # Run evaluations
    results = {p: [] for p in _POLICIES}
    all_rows = []

    for policy_name in _POLICIES:
        print(f"\n  Running: {policy_name}")
        print(_DIV)

        for n in steps:
            subset = all_requests[:n]

            if policy_name == "DDQN":
                m = _run_ddqn_policy(subset, topology, catalogue, orchestrator, prov_interval)
            elif policy_name == "LCA":
                m = _run_heuristic_policy(subset, topology, catalogue, lca_fn, prov_interval, "LCA")
            elif policy_name == "TopDown":
                m = _run_heuristic_policy(subset, topology, catalogue, td_fn, prov_interval, "TopDown")
            elif policy_name == "Greedy":
                m = _run_heuristic_policy(subset, topology, catalogue, greedy_fn, prov_interval, "Greedy")

            results[policy_name].append(m)
            all_rows.append(m)

            print(f"    N={n:>6}  Revenue={m['revenue']:>10,.0f}  Profit={m['profit']:>10,.0f}  "
                  f"Drop={m['drop_rate']*100:>5.1f}%  Local={m['local_rate']*100:>5.1f}%")

    # Save CSV
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "combined_metrics.csv")

    fieldnames = ["policy", "n_requests", "executed", "dropped", "local_rate", "cloud_rate", 
                 "drop_rate", "revenue", "placement_cost", "forwarding_cost", "violation_cost", 
                 "total_cost", "profit"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n  CSV saved: {csv_path}")

    # Generate plots
    print("\n  Generating plots...")
    plot_trend_charts(results, out_dir)
    plot_ddqn_cost_breakdown_bars(results, out_dir)
    plot_ddqn_individual_cost_trends(results, out_dir)
    plot_profit_comparison_bar(results, out_dir)

    print(f"\n{_SEP}")
    print("  Complete! All charts saved in:", out_dir)
    print(f"\n  Generated Charts:")
    print(f"    - Trend charts (4): Revenue, Drop Rate, Cloud Rate, Local Rate")
    print(f"    - DDQN cost breakdown: Bar chart with all cost components at key loads")
    print(f"    - DDQN cost trends (3): Individual cost component trends")
    print(f"    - Profit comparison: Bar chart comparing all policies at max load")
    print(_SEP)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Combined policy comparison with profit analysis")
    p.add_argument("--checkpoint", default=_DEFAULT_CHECKPOINT, help="DDQN checkpoint")
    p.add_argument("--requests", default=_DEFAULT_REQUESTS, help="Request CSV")
    p.add_argument("--services", default=_DEFAULT_SERVICES, help="Service CSV")
    p.add_argument("--steps", type=int, nargs="+", default=_DEFAULT_STEPS, help="Load steps")
    p.add_argument("--prov", type=int, default=PROVISIONING_INTERVAL, help="Provisioning interval")
    p.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    p.add_argument("--out-dir", default=_DEFAULT_OUT_DIR, help="Output directory")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        checkpoint=args.checkpoint,
        requests_csv=args.requests,
        services_csv=args.services,
        steps=sorted(set(args.steps)),
        prov_interval=args.prov,
        seed=args.seed,
        out_dir=args.out_dir,
    )
