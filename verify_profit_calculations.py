# -*- coding: utf-8 -*-
"""
verify_profit_calculations.py
-----------------------------
Verification script to manually check profit calculations from analyze_profit.py.

This script:
1. Loads the same requests and services
2. Manually counts actions, escalations, and violations
3. Computes costs using the same formulas
4. Compares against the CSV output from analyze_profit.py

Usage:
  python verify_profit_calculations.py
  python verify_profit_calculations.py --n-requests 1000
"""

import argparse
import csv
import os
import sys
import pandas as pd
from collections import defaultdict

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from edge_caching.config import (
    NUM_LEVELS, NUM_CHILDREN, NODE_CACHE_CAP,
    PROVISIONING_INTERVAL, RANDOM_SEED,
    LAMBDA_PLACEMENT, LAMBDA_FORWARD, LAMBDA_SLA,
)
from edge_caching.core.service import Service
from edge_caching.network.topology import NetworkTopology
from edge_caching.control.orchestrator import EdgeOrchestrator
from edge_caching.utils.simulation.request_generator import load_requests_from_csv

_SEP = "=" * 90
_DIV = "-" * 90


def load_services(csv_path: str) -> dict:
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


def run_ddqn_and_track(
    requests, 
    topology, 
    catalogue, 
    orchestrator, 
    prov_interval,
    verbose=False
):
    """
    Run DDQN policy and manually track all metrics.

    Returns:
        dict with:
            - executed_count
            - dropped_count
            - insert_count
            - delete_count
            - keep_count
            - peer_escalations
            - parent_escalations
            - cloud_escalations
            - total_escalations
            - revenue
    """
    from edge_caching.core.node import ACTION_INSERT, ACTION_DELETE, ACTION_KEEP
    from edge_caching.agents.ddqn_agent import DDQNAgent

    # Reset world
    topology.prime_cloud_cache(catalogue)
    topology.priority_init(catalogue)
    for node in topology.all_nodes:
        node.reset_interval_stats()
        node.total_revenue = 0.0
    for service in catalogue.values():
        service.reset_interval_counters()

    # Reset orchestrator
    orchestrator._prev_snapshots = {}
    orchestrator._last_provision_sim_time = 0.0
    orchestrator.agent.epsilon = 0.0

    # Tracking counters
    executed = 0
    dropped = 0
    insert_count = 0
    delete_count = 0
    keep_count = 0

    peer_escalations = 0
    parent_escalations = 0
    cloud_escalations = 0

    revenue = 0.0

    prov_calls = 0
    total_prov = max(len(requests) // prov_interval, 1)

    for idx, req in enumerate(requests):
        # Fast process: route the request
        origin_node = topology.node_map[req.origin_node_id]
        origin_node.receive_request(req, catalogue)

        # Track outcomes
        if req.status == "executed":
            executed += 1
            svc = catalogue[req.service_id]
            revenue += svc.revenue * svc.omega

            # Count escalation type
            if req.executed_at_node == req.origin_node_id:
                # Local execution - no escalation
                pass
            elif len(req.hop_path) > 2:
                cloud_escalations += 1
            elif len(req.hop_path) == 2:
                parent_escalations += 1
            elif len(req.hop_path) == 1:
                peer_escalations += 1

        elif req.status == "dropped":
            dropped += 1

        # Slow process: provisioning
        if (idx + 1) % prov_interval == 0:
            prov_calls += 1

            for node in orchestrator.nodes:
                snap = node.get_stats_snapshot()
                for svc_id, service in catalogue.items():
                    state = DDQNAgent.build_state_vector(service, snap)
                    action = orchestrator.agent.select_action(state)

                    if action == ACTION_INSERT:
                        insert_count += 1
                    elif action == ACTION_DELETE:
                        delete_count += 1
                    else:
                        keep_count += 1

                    node.apply_action(action, service)

            # Update stats
            total_demand = max(sum(s.demand_count for s in catalogue.values()), 1)
            for service in catalogue.values():
                service.update_popularity(total_demand)
                service.reset_interval_counters()
            for node in topology.all_nodes:
                node.reset_interval_stats()

    total_escalations = peer_escalations + parent_escalations + cloud_escalations

    return {
        "executed_count": executed,
        "dropped_count": dropped,
        "insert_count": insert_count,
        "delete_count": delete_count,
        "keep_count": keep_count,
        "peer_escalations": peer_escalations,
        "parent_escalations": parent_escalations,
        "cloud_escalations": cloud_escalations,
        "total_escalations": total_escalations,
        "revenue": revenue,
    }


def compute_costs(metrics):
    """Manually compute costs from tracked metrics."""
    # Placement cost = (INSERT + DELETE) * LAMBDA_PLACEMENT
    placement_actions = metrics["insert_count"] + metrics["delete_count"]
    placement_cost = placement_actions * LAMBDA_PLACEMENT

    # Forwarding cost = total_escalations * LAMBDA_FORWARD
    forwarding_cost = metrics["total_escalations"] * LAMBDA_FORWARD

    # Violation cost = dropped_count * LAMBDA_SLA
    violation_cost = metrics["dropped_count"] * LAMBDA_SLA

    # Total cost
    total_cost = placement_cost + forwarding_cost + violation_cost

    # Profit
    profit = metrics["revenue"] - total_cost

    return {
        "placement_cost": placement_cost,
        "forwarding_cost": forwarding_cost,
        "violation_cost": violation_cost,
        "total_cost": total_cost,
        "profit": profit,
    }


def verify_against_csv(expected_metrics, costs, n_requests, csv_path):
    """Compare computed values against CSV output."""
    print("\n" + _SEP)
    print(f"  VERIFICATION: Comparing against {csv_path}")
    print(_DIV)

    # Load the CSV
    df = pd.read_csv(csv_path)

    # Find the row for this number of requests
    row = df[df['n_requests'] == n_requests]

    if row.empty:
        print(f"  WARNING: No data found for n_requests={n_requests} in CSV")
        return False

    row = row.iloc[0]

    # Compare metrics
    comparisons = [
        ("Executed Requests", expected_metrics["executed_count"], row["executed"]),
        ("Dropped Requests", expected_metrics["dropped_count"], row["dropped"]),
        ("INSERT Actions", expected_metrics["insert_count"], row["insert_count"]),
        ("DELETE Actions", expected_metrics["delete_count"], row["delete_count"]),
        ("KEEP Actions", expected_metrics["keep_count"], row["keep_count"]),
        ("Revenue", expected_metrics["revenue"], row["revenue"]),
        ("Placement Cost", costs["placement_cost"], row["placement_cost"]),
        ("Forwarding Cost", costs["forwarding_cost"], row["forwarding_cost"]),
        ("Violation Cost", costs["violation_cost"], row["violation_cost"]),
        ("Total Cost", costs["total_cost"], row["total_cost"]),
        ("Net Profit", costs["profit"], row["profit"]),
    ]

    all_match = True
    print(f"  {'Metric':<25}  {'Computed':<15}  {'From CSV':<15}  {'Match':<10}")
    print(_DIV)

    for name, computed, csv_val in comparisons:
        # Allow small floating point differences
        match = abs(computed - csv_val) < 0.01
        match_str = "YES" if match else "NO"

        if not match:
            all_match = False

        print(f"  {name:<25}  {computed:<15,.2f}  {csv_val:<15,.2f}  {match_str:<10}")

    print(_SEP)

    if all_match:
        print("\n  [OK] ALL METRICS MATCH! Calculations are verified correct.")
    else:
        print("\n  [ERROR] MISMATCH DETECTED! Please review the differences above.")

    return all_match


def detailed_cost_breakdown(metrics, costs):
    """Print detailed breakdown of cost calculations."""
    print("\n" + _SEP)
    print("  DETAILED COST CALCULATION BREAKDOWN")
    print(_DIV)

    print("\n  1. PLACEMENT COST")
    print(f"     Formula: (INSERT + DELETE) x LAMBDA_PLACEMENT")
    print(f"     INSERT actions:  {metrics['insert_count']:>10,}")
    print(f"     DELETE actions:  {metrics['delete_count']:>10,}")
    print(f"     Total actions:   {metrics['insert_count'] + metrics['delete_count']:>10,}")
    print(f"     LAMBDA_PLACEMENT: {LAMBDA_PLACEMENT:>9.4f}")
    print(f"     Placement Cost = {metrics['insert_count'] + metrics['delete_count']:,} x {LAMBDA_PLACEMENT} = {costs['placement_cost']:,.2f}")

    print("\n  2. FORWARDING COST")
    print(f"     Formula: Total_Escalations x LAMBDA_FORWARD")
    print(f"     Peer escalations:   {metrics['peer_escalations']:>10,}")
    print(f"     Parent escalations: {metrics['parent_escalations']:>10,}")
    print(f"     Cloud escalations:  {metrics['cloud_escalations']:>10,}")
    print(f"     Total escalations:  {metrics['total_escalations']:>10,}")
    print(f"     LAMBDA_FORWARD:      {LAMBDA_FORWARD:>9.2f}")
    print(f"     Forwarding Cost = {metrics['total_escalations']:,} x {LAMBDA_FORWARD} = {costs['forwarding_cost']:,.2f}")

    print("\n  3. VIOLATION COST")
    print(f"     Formula: Dropped_Requests x LAMBDA_SLA")
    print(f"     Dropped requests: {metrics['dropped_count']:>10,}")
    print(f"     LAMBDA_SLA:        {LAMBDA_SLA:>9.2f}")
    print(f"     Violation Cost = {metrics['dropped_count']:,} x {LAMBDA_SLA} = {costs['violation_cost']:,.2f}")

    print("\n  4. TOTAL COST")
    print(f"     Placement:  {costs['placement_cost']:>12,.2f}")
    print(f"     Forwarding: {costs['forwarding_cost']:>12,.2f}")
    print(f"     Violation:  {costs['violation_cost']:>12,.2f}")
    print(f"     Total:      {costs['total_cost']:>12,.2f}")

    print("\n  5. NET PROFIT")
    print(f"     Revenue:    {metrics['revenue']:>12,.2f}")
    print(f"     Total Cost: {costs['total_cost']:>12,.2f}")
    print(f"     Profit:     {costs['profit']:>12,.2f}")

    print(_SEP)


def main():
    parser = argparse.ArgumentParser(
        description="Verify profit calculations from analyze_profit.py"
    )
    parser.add_argument("--n-requests", type=int, default=1000,
                       help="Number of requests to verify (default: 1000)")
    parser.add_argument("--checkpoint", type=str,
                       default="checkpoints/ddqn_finetuned_resumed_500epv3.pt",
                       help="DDQN checkpoint file")
    parser.add_argument("--requests", type=str,
                       default="training/data/sample_requests_10000.csv",
                       help="Request trace CSV")
    parser.add_argument("--services", type=str,
                       default="training/data/sample_services.csv",
                       help="Service catalogue CSV")
    parser.add_argument("--csv", type=str,
                       default="results/ddqn_profit_analysis.csv",
                       help="CSV output from analyze_profit.py to verify against")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed per-request information")

    args = parser.parse_args()

    print(_SEP)
    print("  PROFIT CALCULATION VERIFICATION")
    print(_DIV)
    print(f"  Verifying:     {args.n_requests:,} requests")
    print(f"  Checkpoint:    {args.checkpoint}")
    print(f"  Requests CSV:  {args.requests}")
    print(f"  Services CSV:  {args.services}")
    print(f"  Compare with:  {args.csv}")
    print(_SEP)

    # Load services
    catalogue = load_services(args.services)
    print(f"  Loaded {len(catalogue)} services")

    # Build topology
    topology = NetworkTopology(
        num_levels   = NUM_LEVELS,
        num_children = NUM_CHILDREN,
        cache_cap    = NODE_CACHE_CAP,
    )
    topology.prime_cloud_cache(catalogue)
    topology.priority_init(catalogue)
    print(f"  Topology: {NUM_LEVELS} levels, {NUM_CHILDREN} children, {len(topology.all_nodes)} nodes")

    # Load requests
    all_requests = load_requests_from_csv(args.requests, catalogue, topology.node_map)
    print(f"  Loaded {len(all_requests):,} requests")

    # Build DDQN orchestrator
    orchestrator = EdgeOrchestrator(nodes=topology.all_nodes, catalogue=catalogue)
    if os.path.exists(args.checkpoint):
        orchestrator.load_checkpoint(args.checkpoint)
        orchestrator.agent.epsilon = 0.0
        print(f"  DDQN checkpoint loaded (epsilon=0)")
    else:
        print(f"  WARNING: checkpoint not found!")

    # Take subset
    subset = all_requests[:args.n_requests]
    print(f"\n  Running DDQN on first {args.n_requests:,} requests...")

    # Run and track
    metrics = run_ddqn_and_track(
        subset, 
        topology, 
        catalogue, 
        orchestrator, 
        PROVISIONING_INTERVAL,
        args.verbose
    )

    print(f"  Done. Executed: {metrics['executed_count']}, Dropped: {metrics['dropped_count']}")

    # Compute costs
    costs = compute_costs(metrics)

    # Print detailed breakdown
    detailed_cost_breakdown(metrics, costs)

    # Verify against CSV
    if os.path.exists(args.csv):
        verify_against_csv(metrics, costs, args.n_requests, args.csv)
    else:
        print(f"\n  WARNING: CSV file not found: {args.csv}")
        print("  Run analyze_profit.py first to generate the comparison data.")

    print()


if __name__ == "__main__":
    main()
