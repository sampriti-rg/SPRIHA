"""
analyze_profit.py
-----------------
Analyze and visualize DDQN profit breakdown across different request loads.

This script calculates for DDQN policy at multiple load levels (100, 1000, 2000, ... 10000):
  - Revenue from executing requests
  - Placement costs (INSERT/DELETE operations)
  - Forwarding costs (peer, parent, cloud escalations)
  - SLA violation costs (dropped requests)
  - Net Profit = Revenue - (Placement + Forwarding + Violation costs)

Outputs:
  - Console table with detailed breakdown by load
  - Bar chart showing Revenue, Costs, and Profit across load steps
  - CSV file with complete cost/profit analysis

Usage:
  python analyze_profit.py
  python analyze_profit.py --checkpoint checkpoints/ddqn_finetuned_resumed_500epv3.pt
  python analyze_profit.py --steps 100 500 1000 2500 5000 7500 10000
"""

import argparse
import csv
import os
import sys
import random
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from edge_caching.config import (
    NUM_LEVELS, NUM_CHILDREN, NODE_CACHE_CAP,
    PROVISIONING_INTERVAL, RANDOM_SEED, RESULTS_DIR,
    LAMBDA_PLACEMENT, LAMBDA_FORWARD, LAMBDA_SLA,
)
from edge_caching.core.node import EdgeNode, ACTION_INSERT, ACTION_DELETE, ACTION_KEEP
from edge_caching.core.service import Service, generate_service_catalogue
from edge_caching.core.request import Request
from edge_caching.network.topology import NetworkTopology
from edge_caching.control.orchestrator import EdgeOrchestrator
from edge_caching.agents.ddqn_agent import DDQNAgent
from edge_caching.utils.simulation.request_generator import load_requests_from_csv

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_CHECKPOINT = os.path.join(_ROOT, "checkpoints", "ddqn_finetuned_resumed_500epv3.pt")
_DEFAULT_REQUESTS   = os.path.join(_ROOT, "training", "data", "sample_requests_10000.csv")
_DEFAULT_SERVICES   = os.path.join(_ROOT, "training", "data", "sample_services.csv")
_DEFAULT_OUT_CSV    = os.path.join(_ROOT, RESULTS_DIR, "ddqn_profit_analysis.csv")
_DEFAULT_OUT_PLOT   = os.path.join(_ROOT, RESULTS_DIR, "ddqn_profit_comparison.png")
_DEFAULT_STEPS      = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

_TIERS    = ["mission_critical", "high", "standard", "best_effort"]

_SEP = "=" * 90
_DIV = "-" * 90

# Callable type: (node, service, snapshot) -> action int (0/1/2)
ActionFn = Callable[[EdgeNode, Service, dict], int]


# ---------------------------------------------------------------------------
# Service catalogue loader
# ---------------------------------------------------------------------------

def _load_services(csv_path: str) -> Dict[int, Service]:
    """Load service catalogue from CSV."""
    catalogue = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = Service(
                service_id  = int(row["service_id"]),
                size        = float(row.get("size_mb", row.get("size", 0))),
                popularity  = float(row["popularity"]),
                criticality = row["criticality"],
                revenue     = float(row["revenue"]),
            )
            catalogue[s.service_id] = s
    return catalogue


# ---------------------------------------------------------------------------
# World reset
# ---------------------------------------------------------------------------

def _reset_world(topology: NetworkTopology, catalogue: Dict[int, Service]) -> None:
    """Reset topology and service stats for a fresh evaluation run."""
    topology.prime_cloud_cache(catalogue)
    topology.priority_init(catalogue)
    for node in topology.all_nodes:
        node.reset_interval_stats()
        node.total_revenue = 0.0
    for service in catalogue.values():
        service.reset_interval_counters()


# ---------------------------------------------------------------------------
# Tier classifier
# ---------------------------------------------------------------------------

def _classify_tier(req: Request) -> str:
    """Classify request outcome into tier based on where it was executed."""
    if req.status == "dropped":
        return "dropped"
    if req.status != "executed":
        return "unknown"

    # If executed at the origin node, it's local (leaf)
    if req.executed_at_node == req.origin_node_id:
        return "leaf"

    # Check the hop path to determine escalation type
    # If more than 2 hops or executed at cloud, it's cloud escalation
    if len(req.hop_path) > 2:
        return "cloud"

    # 2 hops = parent escalation, 1 hop = peer
    if len(req.hop_path) == 2:
        return "parent"
    elif len(req.hop_path) == 1:
        return "peer"

    return "unknown"


# ---------------------------------------------------------------------------
# Policy implementations
# ---------------------------------------------------------------------------

def greedy_policy(node: EdgeNode, service: Service, snap: dict) -> int:
    """Greedy demand-reactive policy."""
    is_cached = service.service_id in node.cache
    fwd_count = snap.get("forwarded_count", 0)
    exec_count = snap.get("executed", 0)
    served_count = snap.get("served", 0)

    if not is_cached and fwd_count > 0:
        return ACTION_INSERT
    if is_cached and exec_count == 0 and served_count == 0 and fwd_count == 0:
        return ACTION_DELETE
    return ACTION_KEEP


def baseline_policy(node: EdgeNode, service: Service, snap: dict) -> int:
    """Static baseline - never change cache after priority_init."""
    return ACTION_KEEP


def make_ddqn_policy(agent: DDQNAgent) -> ActionFn:
    """Create DDQN policy function."""
    def ddqn_action_fn(node: EdgeNode, service: Service, snap: dict) -> int:
        state = agent.build_state_vector(node, service, snap)
        return agent.select_action(state)
    return ddqn_action_fn


# ---------------------------------------------------------------------------
# Cost tracking runner for heuristic policies
# ---------------------------------------------------------------------------

def _run_heuristic_policy(
    requests      : List[Request],
    topology      : NetworkTopology,
    catalogue     : Dict[int, Service],
    policy_fn     : ActionFn,
    prov_interval : int,
    policy_name   : str,
) -> dict:
    """
    Evaluate a heuristic policy with cost tracking.

    Tracks:
      - Revenue from executed requests
      - Placement costs (INSERT/DELETE actions)
      - Forwarding costs (peer, parent, cloud escalations)
      - Violation costs (dropped requests)
    """
    _reset_world(topology, catalogue)

    tiers           = defaultdict(int)
    revenue_total   = 0.0
    placement_cost  = 0.0
    forwarding_cost = 0.0
    violation_cost  = 0.0

    insert_count    = 0
    delete_count    = 0
    keep_count      = 0

    total = len(requests)

    for idx, req in enumerate(requests):
        # Track request for policy-specific logic
        if hasattr(policy_fn, 'lca_instance'):
            policy_fn.lca_instance.track_request(req.service_id, req.origin_node_id)
        if hasattr(policy_fn, 'td_instance'):
            policy_fn.td_instance.track_request(req.service_id, req.origin_node_id)

        # Fast process: route the request
        topology.node_map[req.origin_node_id].receive_request(req, catalogue)

        tier = _classify_tier(req)
        tiers[tier] += 1

        # Calculate revenue
        if req.status == "executed":
            svc = catalogue[req.service_id]
            rv  = svc.revenue * svc.omega
            revenue_total += rv

        # Calculate forwarding cost
        if tier == "peer":
            forwarding_cost += REWARD_FORWARDING_COST
        elif tier == "parent":
            forwarding_cost += REWARD_FORWARDING_COST
        elif tier == "cloud":
            forwarding_cost += REWARD_FORWARDING_COST

        # Calculate violation cost
        if tier == "dropped":
            violation_cost += REWARD_VIOLATION_PENALTY

        # Slow process: provision every N requests
        if (idx + 1) % prov_interval == 0:
            # Top-Down needs to compute all decisions first
            if hasattr(policy_fn, 'td_instance'):
                policy_fn.td_instance.compute_all_decisions(catalogue)

            for node in topology.all_nodes:
                snap = node.get_stats_snapshot()
                for svc_id, service in catalogue.items():
                    action = policy_fn(node, service, snap)

                    # Track action counts
                    if action == ACTION_INSERT:
                        insert_count += 1
                        placement_cost += LAMBDA_PLACEMENT
                    elif action == ACTION_DELETE:
                        delete_count += 1
                        placement_cost += LAMBDA_PLACEMENT
                    else:
                        keep_count += 1

                    node.apply_action(action, service)

            # Update popularity EMA and reset interval counters
            total_demand = max(sum(s.demand_count for s in catalogue.values()), 1)
            for service in catalogue.values():
                service.update_popularity(total_demand)
                service.reset_interval_counters()
            for node in topology.all_nodes:
                node.reset_interval_stats()

            # Reset policy tracking for next interval
            if hasattr(policy_fn, 'lca_instance'):
                policy_fn.lca_instance.reset_interval()
            if hasattr(policy_fn, 'td_instance'):
                policy_fn.td_instance.reset_interval()

    # Calculate profit
    total_cost = placement_cost + forwarding_cost + violation_cost
    profit = revenue_total - total_cost

    executed = total - tiers["dropped"]

    return {
        "policy"           : policy_name,
        "n_requests"       : total,
        "executed"         : executed,
        "dropped"          : tiers["dropped"],
        "local_rate"       : round(tiers["leaf"]  / max(total, 1), 6),
        "cloud_rate"       : round(tiers["cloud"] / max(total, 1), 6),
        "drop_rate"        : round(tiers["dropped"] / max(total, 1), 6),
        "revenue"          : round(revenue_total, 2),
        "placement_cost"   : round(placement_cost, 2),
        "forwarding_cost"  : round(forwarding_cost, 2),
        "violation_cost"   : round(violation_cost, 2),
        "total_cost"       : round(total_cost, 2),
        "profit"           : round(profit, 2),
        "insert_count"     : insert_count,
        "delete_count"     : delete_count,
        "keep_count"       : keep_count,
    }


# ---------------------------------------------------------------------------
# DDQN policy runner with cost tracking
# ---------------------------------------------------------------------------

def _run_ddqn_policy(
    requests      : List[Request],
    topology      : NetworkTopology,
    catalogue     : Dict[int, Service],
    orchestrator  : EdgeOrchestrator,
    prov_interval : int,
) -> dict:
    """
    Evaluate DDQN policy with cost tracking.

    Tracks same costs as heuristic policies for fair comparison.
    """
    _reset_world(topology, catalogue)

    # Reset orchestrator state
    orchestrator._prev_snapshots           = {}
    orchestrator._last_provision_sim_time  = 0.0
    orchestrator.agent.epsilon             = 0.0

    tiers           = defaultdict(int)
    revenue_total   = 0.0
    placement_cost  = 0.0
    forwarding_cost = 0.0
    violation_cost  = 0.0

    insert_count    = 0
    delete_count    = 0
    keep_count      = 0

    total = len(requests)
    prov_calls = 0
    total_prov = max(total // prov_interval, 1)

    for idx, req in enumerate(requests):
        # Fast process
        topology.node_map[req.origin_node_id].receive_request(req, catalogue)

        tier = _classify_tier(req)
        tiers[tier] += 1

        # Calculate revenue
        if req.status == "executed":
            svc = catalogue[req.service_id]
            rv  = svc.revenue * svc.omega
            revenue_total += rv

        # Calculate forwarding cost
        if tier == "peer":
            forwarding_cost += LAMBDA_FORWARD
        elif tier == "parent":
            forwarding_cost += LAMBDA_FORWARD
        elif tier == "cloud":
            forwarding_cost += LAMBDA_FORWARD

        # Calculate violation cost
        if tier == "dropped":
            violation_cost += LAMBDA_SLA

        # Slow process - track actions during provision
        if (idx + 1) % prov_interval == 0:
            prov_calls += 1

            # Track actions taken by DDQN
            action_counts = {"insert": 0, "delete": 0, "keep": 0}

            # Manually run provision to track actions
            for node in orchestrator.nodes:
                snap = node.get_stats_snapshot()
                for svc_id, service in catalogue.items():
                    state = DDQNAgent.build_state_vector(service, snap)
                    action = orchestrator.agent.select_action(state)

                    if action == ACTION_INSERT:
                        action_counts["insert"] += 1
                        placement_cost += LAMBDA_PLACEMENT
                    elif action == ACTION_DELETE:
                        action_counts["delete"] += 1
                        placement_cost += LAMBDA_PLACEMENT
                    else:
                        action_counts["keep"] += 1

                    node.apply_action(action, service)

            insert_count += action_counts["insert"]
            delete_count += action_counts["delete"]
            keep_count += action_counts["keep"]

            # Update popularity and reset stats
            total_demand = max(sum(s.demand_count for s in catalogue.values()), 1)
            for service in catalogue.values():
                service.update_popularity(total_demand)
                service.reset_interval_counters()
            for node in topology.all_nodes:
                node.reset_interval_stats()

    # Calculate profit
    total_cost = placement_cost + forwarding_cost + violation_cost
    profit = revenue_total - total_cost

    executed = total - tiers["dropped"]

    return {
        "policy"           : "DDQN",
        "n_requests"       : total,
        "executed"         : executed,
        "dropped"          : tiers["dropped"],
        "local_rate"       : round(tiers["leaf"]  / max(total, 1), 6),
        "cloud_rate"       : round(tiers["cloud"] / max(total, 1), 6),
        "drop_rate"        : round(tiers["dropped"] / max(total, 1), 6),
        "revenue"          : round(revenue_total, 2),
        "placement_cost"   : round(placement_cost, 2),
        "forwarding_cost"  : round(forwarding_cost, 2),
        "violation_cost"   : round(violation_cost, 2),
        "total_cost"       : round(total_cost, 2),
        "profit"           : round(profit, 2),
        "insert_count"     : insert_count,
        "delete_count"     : delete_count,
        "keep_count"       : keep_count,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _plot_profit_comparison(results: List[dict], out_path: str) -> None:
    """
    Create three-panel visualization showing DDQN profit breakdown across load steps.

    Panel 1: Revenue, Costs, Profit bars with Profit Margin line
    Panel 2: Placement & Forwarding Cost trends
    Panel 3: SLA Violation trend
    """
    fig = plt.figure(figsize=(24, 7))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Extract data
    n_requests = [r["n_requests"] for r in results]
    revenues = [r["revenue"] for r in results]
    placement_costs = [r["placement_cost"] for r in results]
    forwarding_costs = [r["forwarding_cost"] for r in results]
    violation_costs = [r["violation_cost"] for r in results]
    total_costs = [r["total_cost"] for r in results]
    profits = [r["profit"] for r in results]
    dropped = [r["dropped"] for r in results]

    # Calculate profit margins (profit / revenue * 100)
    profit_margins = [(p / r * 100) if r > 0 else 0 for p, r in zip(profits, revenues)]

    x = np.arange(len(n_requests))
    width = 0.2

    # =========================================================================
    # PANEL 1: Revenue, Profit, and All Costs as Bars with Profit Margin Line
    # =========================================================================

    bars1 = ax1.bar(x - 1.5*width, revenues, width, label='Revenue', 
                    color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=0.8)
    bars2 = ax1.bar(x - 0.5*width, profits, width, label='Net Profit', 
                    color='#3498db', alpha=0.85, edgecolor='black', linewidth=0.8)
    bars3 = ax1.bar(x + 0.5*width, placement_costs, width, label='Placement Cost', 
                    color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=0.8)
    bars4 = ax1.bar(x + 1.5*width, forwarding_costs, width, label='Forwarding Cost', 
                    color='#e67e22', alpha=0.85, edgecolor='black', linewidth=0.8)

    # Add value labels on bars
    for bars, values in [(bars1, revenues), (bars2, profits), 
                         (bars3, placement_costs), (bars4, forwarding_costs)]:
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            if val > max(revenues) * 0.02:  # Only show significant values
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(revenues) * 0.01,
                        f'{val:,.0f}',
                        ha='center', va='bottom',
                        fontsize=7, fontweight='bold', alpha=0.8)

    # Add profit margin line on secondary y-axis
    #ax1_twin = ax1.twinx()
    #line = ax1_twin.plot(x, profit_margins, color='#9b59b6', marker='D', linewidth=2.5, 
    #                      markersize=7, label='Profit Margin (%)', zorder=10)

    # Add profit margin value labels
    #for i, (xi, pm) in enumerate(zip(x, profit_margins)):
    #    ax1_twin.text(xi, pm + 1.5, f'{pm:.1f}%',
    #                 ha='center', va='bottom',
    #                 fontsize=8, fontweight='bold', color='#9b59b6',
    #                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
    #                          edgecolor='#9b59b6', alpha=0.8))

    ax1.set_xlabel('Number of Requests', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Amount (Units)', fontsize=12, fontweight='bold')
    #ax1_twin.set_ylabel('Profit Margin (%)', fontsize=12, fontweight='bold', color='#9b59b6')
    #ax1_twin.tick_params(axis='y', labelcolor='#9b59b6')
    ax1.set_title('Panel 1: Revenue, Profit & Costs with Profit Margin', 
                  fontsize=14, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{n:,}' for n in n_requests], rotation=45, ha='right', fontsize=9)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    #lines2, labels2 = ax1_twin.get_legend_handles_labels()
    #ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    ax1.legend(lines1, labels1, loc='upper left', fontsize=9)


    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Set profit margin y-axis limits
    max_margin = max(profit_margins) if profit_margins else 100
    #ax1_twin.set_ylim(0, max_margin + 10)

    # =========================================================================
    # PANEL 2: Placement & Forwarding Cost Trends
    # =========================================================================

    # Plot trends
    line1 = ax2.plot(x, placement_costs, color='#e74c3c', marker='o', linewidth=2.5, 
                     markersize=8, label='Placement Cost', zorder=10)
    line2 = ax2.plot(x, forwarding_costs, color='#e67e22', marker='s', linewidth=2.5, 
                     markersize=8, label='Forwarding Cost', zorder=10)

    # Add value labels
    for i, (xi, pc, fc) in enumerate(zip(x, placement_costs, forwarding_costs)):
        # Placement cost label
        ax2.text(xi, pc, f'{pc:,.0f}',
                ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='#e74c3c',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='#e74c3c', alpha=0.8))

        # Forwarding cost label
        ax2.text(xi, fc, f'{fc:,.0f}',
                ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='#e67e22',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='#e67e22', alpha=0.8))

    ax2.set_xlabel('Number of Requests', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cost (Units)', fontsize=12, fontweight='bold')
    ax2.set_title('Panel 2: Placement & Forwarding Cost Trends', 
                  fontsize=14, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{n:,}' for n in n_requests], rotation=45, ha='right', fontsize=9)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Fill area under curves for better visualization
    ax2.fill_between(x, placement_costs, alpha=0.2, color='#e74c3c')
    ax2.fill_between(x, forwarding_costs, alpha=0.2, color='#e67e22')

    # =========================================================================
    # PANEL 3: SLA Violation Trend
    # =========================================================================

    # Plot violation trend
    line3 = ax3.plot(x, violation_costs, color='#9b59b6', marker='^', linewidth=3, 
                     markersize=10, label='Violation Cost', zorder=10)

    # Secondary y-axis for dropped count
    #ax3_twin = ax3.twinx()
    #line4 = ax3_twin.plot(x, dropped, color='#c0392b', marker='X', linewidth=2.5, 
    #                      markersize=9, label='Dropped Requests', 
    #                      linestyle='--', zorder=9)

    # Add value labels for violation costs
    for i, (xi, vc) in enumerate(zip(x, violation_costs)):
        ax3.text(xi, vc, f'{vc:,.0f}',
                ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#9b59b6',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         edgecolor='#9b59b6', alpha=0.9, linewidth=2))

    # Add value labels for dropped requests
    #for i, (xi, d) in enumerate(zip(x, dropped)):
    #    ax3_twin.text(xi, d, f'{d}',
    #                 ha='center', va='top',
    #                 fontsize=8, fontweight='bold', color='#c0392b',
    #                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
    #                          edgecolor='#c0392b', alpha=0.8))

    ax3.set_xlabel('Number of Requests', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Violation Cost (Units)', fontsize=12, fontweight='bold', color='#9b59b6')
    ax3.tick_params(axis='y', labelcolor='#9b59b6')
    #ax3_twin.set_ylabel('Dropped Requests (Count)', fontsize=12, fontweight='bold', color='#c0392b')
    #ax3_twin.tick_params(axis='y', labelcolor='#c0392b')
    ax3.set_title('Panel 3: SLA Violation Trend', 
                  fontsize=14, fontweight='bold', pad=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{n:,}' for n in n_requests], rotation=45, ha='right', fontsize=9)

    # Combine legends
    lines3a, labels3a = ax3.get_legend_handles_labels()
    #lines3b, labels3b = ax3_twin.get_legend_handles_labels()
    #ax3.legend(lines3a + lines3b, labels3a + labels3b, loc='upper left', fontsize=10)
    ax3.legend(lines3a, labels3a, loc='upper left', fontsize=10)

    ax3.grid(True, alpha=0.3, linestyle='--')

    # Fill area under violation cost curve
    ax3.fill_between(x, violation_costs, alpha=0.3, color='#9b59b6')

    # Add warning threshold line if violations are high
    max_violations = max(violation_costs) if violation_costs else 0
    if max_violations > 0:
        avg_violations = sum(violation_costs) / len(violation_costs)
        ax3.axhline(y=avg_violations, color='red', linestyle=':', linewidth=2, 
                   alpha=0.5, label=f'Avg: {avg_violations:.0f}')

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"  Profit comparison plot saved -> {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def _print_summary_table(results: List[dict]) -> None:
    """Print detailed profit breakdown table."""
    print("\n" + _SEP)
    print("  DDQN PROFIT ANALYSIS SUMMARY")
    print(_DIV)

    # Header
    print(f"  {'Requests':<12}  {'Executed':<10}  {'Dropped':<10}  "
          f"{'Revenue':<12}  {'Total Cost':<12}  {'Net Profit':<12}  {'Margin %':<10}")
    print(_DIV)

    for r in results:
        profit_margin = (r['profit'] / r['revenue'] * 100) if r['revenue'] > 0 else 0
        print(
            f"  {r['n_requests']:<12,}  "
            f"{r['executed']:<10,}  "
            f"{r['dropped']:<10,}  "
            f"{r['revenue']:<12,.0f}  "
            f"{r['total_cost']:<12,.0f}  "
            f"{r['profit']:<12,.0f}  "
            f"{profit_margin:<10.1f}"
        )

    print(_SEP)

    # Cost breakdown table
    print("\n  COST BREAKDOWN BY LOAD")
    print(_DIV)
    print(f"  {'Requests':<12}  {'Placement':<12}  {'Forwarding':<12}  "
          f"{'Violation':<12}  {'Total':<12}")
    print(_DIV)

    for r in results:
        print(
            f"  {r['n_requests']:<12,}  "
            f"{r['placement_cost']:<12,.2f}  "
            f"{r['forwarding_cost']:<12,.2f}  "
            f"{r['violation_cost']:<12,.2f}  "
            f"{r['total_cost']:<12,.2f}"
        )

    print(_SEP)

    # Action counts breakdown
    print("\n  PROVISIONING ACTION COUNTS BY LOAD")
    print(_DIV)
    print(f"  {'Requests':<12}  {'INSERT':<10}  {'DELETE':<10}  {'KEEP':<10}  {'Total':<10}")
    print(_DIV)

    for r in results:
        total_actions = r["insert_count"] + r["delete_count"] + r["keep_count"]
        print(
            f"  {r['n_requests']:<12,}  "
            f"{r['insert_count']:<10,}  "
            f"{r['delete_count']:<10,}  "
            f"{r['keep_count']:<10,}  "
            f"{total_actions:<10,}"
        )

    print(_SEP)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    checkpoint    : str       = _DEFAULT_CHECKPOINT,
    requests_csv  : str       = _DEFAULT_REQUESTS,
    services_csv  : str       = _DEFAULT_SERVICES,
    steps         : List[int] = _DEFAULT_STEPS,
    prov_interval : int       = PROVISIONING_INTERVAL,
    seed          : int       = RANDOM_SEED,
    out_csv       : str       = _DEFAULT_OUT_CSV,
    out_plot      : str       = _DEFAULT_OUT_PLOT,
) -> None:

    random.seed(seed)
    np.random.seed(seed)

    print(_SEP)
    print("  DDQN PROFIT ANALYSIS: Load-Step Evaluation")
    print(_DIV)
    print(f"  Checkpoint    : {checkpoint}")
    print(f"  Requests      : {requests_csv}")
    print(f"  Services      : {services_csv}")
    print(f"  Load steps    : {steps}")
    print(f"  Prov interval : {prov_interval}")
    print(f"  Random seed   : {seed}")
    print(f"  Output CSV    : {out_csv}")
    print(f"  Output plot   : {out_plot}")
    print(_SEP)

    # Load service catalogue
    if services_csv and os.path.exists(services_csv):
        catalogue = _load_services(services_csv)
        print(f"  Loaded {len(catalogue)} services from {services_csv}")
    else:
        catalogue = {s.service_id: s for s in generate_service_catalogue(50, seed=seed)}
        print(f"  Generated {len(catalogue)} random services (no CSV found)")

    # Build topology
    topology = NetworkTopology(
        num_levels   = NUM_LEVELS,
        num_children = NUM_CHILDREN,
        cache_cap    = NODE_CACHE_CAP,
    )
    topology.prime_cloud_cache(catalogue)
    topology.priority_init(catalogue)
    print(f"  Topology      : {NUM_LEVELS} levels, {NUM_CHILDREN} children, "
          f"{len(topology.all_nodes)} nodes, {len(topology.leaf_nodes)} leaf nodes")

    # Load request trace
    if not os.path.exists(requests_csv):
        print(f"\n  ERROR: request trace not found: {requests_csv}")
        sys.exit(1)

    all_requests = load_requests_from_csv(requests_csv, catalogue, topology.node_map)
    n_total = len(all_requests)
    print(f"  Loaded        : {n_total} requests")

    # Clamp steps to the available request count
    steps = sorted(set(min(n, n_total) for n in steps))
    print(f"  Effective steps: {steps}\n")

    # Build DDQN orchestrator
    orchestrator = EdgeOrchestrator(nodes=topology.all_nodes, catalogue=catalogue)
    if os.path.exists(checkpoint):
        orchestrator.load_checkpoint(checkpoint)
        orchestrator.agent.epsilon = 0.0
        print(f"  DDQN checkpoint loaded  ->  epsilon=0 (greedy policy)")
    else:
        orchestrator.agent.epsilon = 0.0
        print(f"  WARNING: checkpoint not found at {checkpoint}")
        print(f"           Running DDQN with untrained (random-init) weights")
    print()

    # Run DDQN at each load step
    results = []

    print(f"  Running DDQN policy at different load levels...")
    print(_DIV)

    for n in steps:
        subset = all_requests[:n]

        m = _run_ddqn_policy(
            requests      = subset,
            topology      = topology,
            catalogue     = catalogue,
            orchestrator  = orchestrator,
            prov_interval = prov_interval,
        )

        results.append(m)

        print(
            f"    N={n:>6,}  "
            f"Exec={m['executed']:>6,}  "
            f"Drop={m['dropped']:>5,}  "
            f"Rev={m['revenue']:>10,.0f}  "
            f"Cost={m['total_cost']:>9,.2f}  "
            f"Profit={m['profit']:>10,.0f}"
        )

    print()

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "policy", "n_requests", "executed", "dropped",
            "local_rate", "cloud_rate", "drop_rate",
            "revenue", "placement_cost", "forwarding_cost", "violation_cost",
            "total_cost", "profit",
            "insert_count", "delete_count", "keep_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"  Results CSV saved -> {out_csv}")

    # Plot comparison
    _plot_profit_comparison(results, out_plot)

    # Print summary table
    _print_summary_table(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="analyze_profit.py",
        description="Analyze DDQN profit across different request loads with cost breakdown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python analyze_profit.py\n\n"
            "  python analyze_profit.py \\\n"
            "      --checkpoint checkpoints/ddqn_finetuned_resumed_500epv3.pt\n\n"
            "  python analyze_profit.py \\\n"
            "      --steps 100 500 1000 2500 5000 7500 10000 \\\n"
            "      --out-plot results/ddqn_profit.png\n"
        ),
    )
    p.add_argument("--checkpoint", type=str, default=_DEFAULT_CHECKPOINT,
                   help=f"DDQN checkpoint .pt file")
    p.add_argument("--requests", type=str, default=_DEFAULT_REQUESTS,
                   help=f"Request trace CSV")
    p.add_argument("--services", type=str, default=_DEFAULT_SERVICES,
                   help=f"Service catalogue CSV")
    p.add_argument("--steps", type=int, nargs='+', default=_DEFAULT_STEPS,
                   help=f"Request load levels to evaluate (default: {_DEFAULT_STEPS})")
    p.add_argument("--prov", type=int, default=PROVISIONING_INTERVAL,
                   help=f"Provisioning interval in requests (default: {PROVISIONING_INTERVAL})")
    p.add_argument("--seed", type=int, default=RANDOM_SEED,
                   help=f"Random seed (default: {RANDOM_SEED})")
    p.add_argument("--out-csv", type=str, default=_DEFAULT_OUT_CSV,
                   help=f"Output CSV path")
    p.add_argument("--out-plot", type=str, default=_DEFAULT_OUT_PLOT,
                   help=f"Output PNG path")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        checkpoint    = args.checkpoint,
        requests_csv  = args.requests,
        services_csv  = args.services,
        steps         = args.steps,
        prov_interval = args.prov,
        seed          = args.seed,
        out_csv       = args.out_csv,
        out_plot      = args.out_plot,
    )
