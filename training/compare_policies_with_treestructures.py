"""
compare_policies.py
-------------------
Side-by-side performance comparison  cache provisioning policies
run against the same request trace but with different tree structure and corresponding trained models and relevant sample_requests.

Policies
--------
  DDQN     : Trained model (checkpoint), epsilon=0 pure greedy policy.
             The Q-network observes a 10-element state vector per
             (service, node) pair and outputs the learned INSERT / DELETE /
             KEEP action.  No further learning occurs during evaluation
             (replay buffer starts empty so is_ready() stays False).


Metrics
-------
  drop_rate           Deadline violations / total
  total_revenue       sum(revenue * omega) for all executed requests
Usage
-----
  python compare_policies.py

  python compare_policies.py \\
      --checkpoint checkpoints/ddqn_finetune_v4.pt \\
      --steps 100 500 1000 2000 5000 10000

  python compare_policies.py \\
      --checkpoint checkpoints/ddqn_finetuned.pt \\
      --requests   training/data/sample_requests.csv \\
      --services   training/data/sample_services.csv \\
      --steps      100 250 500 1000 2000 5000 10000 \\
      --out-csv    results/policy_comparison.csv \\
      --out-plot   results/policy_comparison.png

Outputs
-------
  results/policy_comparison.csv   All three policies x all load levels
  results/policy_comparison.png   6-panel comparison dashboard


Execution Command
python compare_policies_All.py --metric local --out-plot results/local_exec.png
python compare_policies_All.py --metric cloud --out-plot results/cloud_esc.png
python compare_policies_All.py --metric drop --out-plot results/drop_rate.png
python compare_policies_All.py --metric revenue --out-plot results/revenue.png
"""

import argparse
import csv
import os
import sys
import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_ROOT)
sys.path.insert(0, _PARENT)

from edge_caching.config import (
    NUM_LEVELS, NUM_CHILDREN, NODE_CACHE_CAP,
    PROVISIONING_INTERVAL, RANDOM_SEED, RESULTS_DIR,
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

_DEFAULT_CHECKPOINT = os.path.join(_ROOT, "checkpoints", "ddqn_finetuned_resumed_500ep.pt")
_DEFAULT_REQUESTS   = os.path.join(_ROOT, "training", "data", "sample_requests_10000_3_3.csv")
_DEFAULT_SERVICES   = os.path.join(_ROOT, "training", "data", "sample_services.csv")
_DEFAULT_OUT_CSV    = os.path.join(_ROOT, RESULTS_DIR, "policy_comparison.csv")
_DEFAULT_OUT_PLOT   = os.path.join(_ROOT, RESULTS_DIR, "policy_comparison.png")
_DEFAULT_STEPS      = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
_DEFAULT_NUM_LEVELS = NUM_LEVELS
_DEFAULT_NUM_CHILDREN = NUM_CHILDREN 
_TIERS    = ["mission_critical", "high", "standard", "best_effort"]
_POLICIES = ["DDQN"]

_SEP = "=" * 80
_DIV = "-" * 80

_CSV_COLUMNS = [
    "policy", "n_requests",
    "hit_rate_mean", "hit_rate_std",
    "local_rate", "cloud_rate", "drop_rate",
    "total_revenue",
    "rev_mission_critical", "rev_high", "rev_standard", "rev_best_effort",
    "rev_per_request", "executed", "dropped",
]

# Callable type: (node, service, snapshot) -> action int (0/1/2)
ActionFn = Callable[[EdgeNode, Service, dict], int]


# ---------------------------------------------------------------------------
# Service catalogue loader
# ---------------------------------------------------------------------------

def _load_services(csv_path: str) -> Dict[int, Service]:
    catalogue: Dict[int, Service] = {}
    with open(csv_path, newline="", encoding="utf-8-sig", errors="replace") as f:
        for row in csv.DictReader(f):
            svc = Service(
                service_id      = int(row["service_id"]),
                size            = int(row["size"]),
                popularity      = float(row["popularity"]),
                criticality     = row["criticality"].strip(),
                revenue         = float(row["revenue"]),
                init_cpu_cycles = float(row["init_cpu_cycles"])
                                  if "init_cpu_cycles" in row and row["init_cpu_cycles"]
                                  else 50_000,
            )
            catalogue[svc.service_id] = svc
    return catalogue


# ---------------------------------------------------------------------------
# World reset -- identical starting state for every (policy, load-level) run
# ---------------------------------------------------------------------------

def _reset_world(topology: NetworkTopology, catalogue: Dict[int, Service]) -> None:
    """
    Wipe all node caches, EDF queues, counters, and revenue accumulators,
    then re-seed with priority_init so every run begins from the same state.
    """
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
# Execution-tier classifier
# ---------------------------------------------------------------------------

def _classify_tier(req: Request) -> str:
    """Classify where a completed request was ultimately handled."""
    if req.status == "dropped":
        return "dropped"
    if req.executed_at_node is None:
        return "unknown"
    if req.executed_at_node == req.origin_node_id:
        return "leaf"
    if req.executed_at_node == "CLOUD":
        return "cloud"
    for _, evt in req.node_log:
        if evt == "cache_miss_peer":
            return "peer"
        if evt in ("cache_miss_parent", "cache_miss_cloud"):
            return "parent"
    return "parent"


# ---------------------------------------------------------------------------
# Policy action functions
# ---------------------------------------------------------------------------

def make_ddqn_policy(agent: DDQNAgent) -> ActionFn:
    """
    Return a closure over the trained DDQN agent.

    At each provision cycle the 10-element state vector is built from
    (service, node snapshot) and fed through the online Q-network.
    argmax over the three Q-values gives the action (epsilon=0).
    """
    def _policy(node: EdgeNode, service: Service, snap: dict) -> int:
        state = DDQNAgent.build_state_vector(service, snap)
        return agent.select_action(state)
    return _policy

# ---------------------------------------------------------------------------
# DDQN policy runner (reuses EdgeOrchestrator.provision())
# ---------------------------------------------------------------------------

def _run_ddqn_policy(
    requests      : List[Request],
    topology      : NetworkTopology,
    catalogue     : Dict[int, Service],
    orchestrator  : EdgeOrchestrator,
    prov_interval : int,
) -> dict:
    """
    Evaluate the DDQN policy using EdgeOrchestrator.provision().

    epsilon is set to 0 before calling so all actions are greedy argmax.
    The replay buffer starts empty so memory.is_ready() stays False and
    no gradient updates occur -- this is pure inference.

    The orchestrator's internal _prev_snapshots and _last_provision_sim_time
    are reset at the start of each run so state doesn't bleed between load
    levels.
    """
    _reset_world(topology, catalogue)

    # Reset orchestrator state for a clean evaluation run
    orchestrator._prev_snapshots           = {}
    orchestrator._last_provision_sim_time  = 0.0
    orchestrator.agent.epsilon             = 0.0

    tiers           = defaultdict(int)
    revenue_total   = 0.0
    revenue_by_tier = {t: 0.0 for t in _TIERS}
    executed_flags  : List[int] = []
    total           = len(requests)
    prov_calls      = 0
    total_prov      = max(total // prov_interval, 1)

    for idx, req in enumerate(requests):
        # ?? Fast process ?????????????????????????????????????????????
        topology.node_map[req.origin_node_id].receive_request(req, catalogue)

        tier = _classify_tier(req)
        tiers[tier] += 1
        executed_flags.append(1 if req.status == "executed" else 0)

        if req.status == "executed":
            svc = catalogue[req.service_id]
            rv  = svc.revenue * svc.omega
            revenue_total                    += rv
            revenue_by_tier[svc.criticality] += rv

        # ?? Slow process ?????????????????????????????????????????????
        if (idx + 1) % prov_interval == 0:
            prov_calls += 1
            orchestrator.provision(
                episode_done = (prov_calls == total_prov),
                sim_time_ms  = req.arrival_time,
            )

    # Finalise episode counters (epsilon decay etc.) -- harmless for eval
    executed = sum(executed_flags)
    dropped  = tiers["dropped"]
    orchestrator.end_episode(
        episode_reward   = 0.0,
        total_executed   = executed,
        total_requests   = total,
        total_forwarded  = tiers["peer"] + tiers["parent"] + tiers["cloud"],
        total_violations = dropped,
        avg_loss         = 0.0,
    )

    return _build_metrics(
        "DDQN", total, executed_flags, tiers,
        revenue_total, revenue_by_tier,
    )


# ---------------------------------------------------------------------------
# Shared metric builder
# ---------------------------------------------------------------------------

def _build_metrics(
    policy_name     : str,
    total           : int,
    executed_flags  : List[int],
    tiers           : dict,
    revenue_total   : float,
    revenue_by_tier : dict,
) -> dict:
    """Compute rolling hit-rate stats and package all metrics into a dict."""
    executed = sum(executed_flags)
    dropped  = tiers["dropped"]

    # Rolling hit rate (window = 50 requests)
    window    = 50
    flags_arr = np.array(executed_flags, dtype=float)
    if len(flags_arr) >= window:
        rolling = np.convolve(flags_arr, np.ones(window) / window, mode="valid")
        hr_mean = float(rolling.mean())
        hr_std  = float(rolling.std())
    else:
        hr_mean = float(flags_arr.mean()) if len(flags_arr) > 0 else 0.0
        hr_std  = 0.0

    return {
        "policy"               : policy_name,
        "n_requests"           : total,
        "hit_rate_mean"        : round(hr_mean,                                         6),
        "hit_rate_std"         : round(hr_std,                                          6),
        "local_rate"           : round(tiers["leaf"]  / max(total, 1),                 6),
        "cloud_rate"           : round(tiers["cloud"] / max(total, 1),                 6),
        "drop_rate"            : round(dropped        / max(total, 1),                 6),
        "total_revenue"        : round(revenue_total,                                   4),
        "rev_mission_critical" : round(revenue_by_tier.get("mission_critical", 0.0),   4),
        "rev_high"             : round(revenue_by_tier.get("high",             0.0),   4),
        "rev_standard"         : round(revenue_by_tier.get("standard",         0.0),   4),
        "rev_best_effort"      : round(revenue_by_tier.get("best_effort",      0.0),   4),
        "rev_per_request"      : round(revenue_total / max(executed, 1),               6),
        "executed"             : executed,
        "dropped"              : dropped,
    }



# ---------------------------------------------------------------------------
# Console summary helpers
# ---------------------------------------------------------------------------

def _print_per_policy_table(results: Dict[str, List[dict]], steps: List[int]) -> None:
    """Print a per-load-level table for each policy."""
    print(
        f"\n  {'Policy':<10}  {'N':>6}  "
        f"{'HitMean%':>9}  {'HitStd%':>8}  "
        f"{'Local%':>7}  {'Cloud%':>7}  {'Drop%':>6}  "
        f"{'Revenue':>10}  {'Rev/Req':>8}"
    )
    print(_DIV)
    for policy_name in _POLICIES:
        for m in results.get(policy_name, []):
            print(
                f"  {policy_name:<10}  {m['n_requests']:>6}  "
                f"{m['hit_rate_mean']*100:>8.2f}%  "
                f"{m['hit_rate_std']*100:>7.2f}%  "
                f"{m['local_rate']*100:>6.2f}%  "
                f"{m['cloud_rate']*100:>6.2f}%  "
                f"{m['drop_rate']*100:>5.2f}%  "
                f"{m['total_revenue']:>10.2f}  "
                f"{m['rev_per_request']:>8.4f}"
            )
        print()


def _print_summary_table(results: Dict[str, List[dict]], steps: List[int]) -> None:
    """Print DDQN deltas vs Top-Down at the maximum load level."""
    print(_SEP)
    print(f"  SUMMARY AT MAXIMUM LOAD  (N = {steps[-1]})")
    print(f"  DDQN delta shows improvement (+) or regression (-) vs Top-Down baseline")
    print(_DIV)

    col_metrics = [
        ("Local exec rate",     "local_rate",           True),
        ("Cloud escalation",    "cloud_rate",           True),
        ("Drop rate",           "drop_rate",            True),
        ("Total revenue",       "total_revenue",        False),
    ]

    last = {p: results[p][-1] for p in _POLICIES if results.get(p)}

    # Print header with all policies
    header = f"  {'Metric':<24}"
    for p in _POLICIES:
        if p in last:
            header += f"  {p:>10}"
    print(header)
    print(_DIV)

    for label, col, pct in col_metrics:
        row = f"  {label:<24}"
        for p in _POLICIES:
            if p in last:
                val = last[p][col]
                if pct:
                    row += f"  {val*100:>9.2f}%"
                else:
                    row += f"  {val:>10,.2f}"
        print(row)
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
    metric        : str       = "all",
    out_csv       : str       = _DEFAULT_OUT_CSV,
    out_plot      : str       = _DEFAULT_OUT_PLOT,
    number_levels    : int       = _DEFAULT_NUM_LEVELS,
    number_children  : int       = _DEFAULT_NUM_CHILDREN
) -> None:

    random.seed(seed)
    np.random.seed(seed)

    print(_SEP)
    print("  Policy Comparison: DDQN vs LCA vs TopDown vs Greedy")
    print(_DIV)
    print(f"  Checkpoint    : {checkpoint}")
    print(f"  Requests      : {requests_csv}")
    print(f"  Services      : {services_csv}")
    print(f"  Load steps    : {steps}")
    print(f"  Prov interval : {prov_interval}")
    print(f"  Metric        : {metric}")
    print(f"  Output CSV    : {out_csv}")
    print(f"  Output plot   : {out_plot}")
    print(_SEP)

    # ?? Build service catalogue ???????????????????????????????????????????
    if services_csv and os.path.exists(services_csv):
        catalogue = _load_services(services_csv)
        print(f"  Loaded {len(catalogue)} services from {services_csv}")
    else:
        catalogue = {
            s.service_id: s
            for s in generate_service_catalogue(50, seed=seed)
        }
        print(f"  Generated {len(catalogue)} random services (no CSV found)")

    # ?? Build topology ????????????????????????????????????????????????????
    topology = NetworkTopology(
        num_levels   = number_levels,
        num_children = number_children,
        cache_cap    = NODE_CACHE_CAP,
    )
    topology.prime_cloud_cache(catalogue)
    topology.priority_init(catalogue)
    
    # Validate tree structure - count nodes at each level
    nodes_per_level = defaultdict(list)
    nodes_per_level[0].append(topology.cloud_root.node_id)
    for node in topology.all_nodes:
        nodes_per_level[node.level].append(node.node_id)
    
    print(f"  Topology      : {number_levels} levels, {number_children} children, "
          f"{len(topology.all_nodes)} edge nodes, "
          f"{len(topology.leaf_nodes)} leaf nodes")
    print(f"\n  Tree Structure Validation:")
    for level in sorted(nodes_per_level.keys()):
        node_ids = nodes_per_level[level]
        print(f"    Level {level}: {len(node_ids)} node(s) - {', '.join(sorted(node_ids))}")
    
    # Verify expected structure
    expected_nodes_at_level = {0: 1}
    for i in range(1, number_levels + 1):
        expected_nodes_at_level[i] = number_children ** i
    
    expected_total_edge = sum(number_children ** i for i in range(1, number_levels + 1))
    expected_leaf = number_children ** number_levels
    
    print(f"\n  Expected vs Actual for {number_levels}-level, {number_children}-children tree:")
    for level in range(number_levels + 1):
        actual = len(nodes_per_level[level])
        expected = expected_nodes_at_level[level]
        status = "✓" if actual == expected else "✗"
        print(f"    Level {level}: Expected {expected:>3}, Got {actual:>3} {status}")
    print(f"    Total edge nodes: Expected {expected_total_edge:>3}, Got {len(topology.all_nodes):>3} {'✓' if len(topology.all_nodes) == expected_total_edge else '✗'}")
    print(f"    Leaf nodes:       Expected {expected_leaf:>3}, Got {len(topology.leaf_nodes):>3} {'✓' if len(topology.leaf_nodes) == expected_leaf else '✗'}")
    print()

    # ?? Load request trace ????????????????????????????????????????????????
    if not os.path.exists(requests_csv):
        print(f"\n  ERROR: request trace not found: {requests_csv}")
        print("  Generate one with:  python generate_request_data.py")
        sys.exit(1)

    all_requests = load_requests_from_csv(requests_csv, catalogue, topology.node_map)
    n_total = len(all_requests)
    print(f"  Loaded        : {n_total} requests")
    
    # Validate request node IDs match topology
    if n_total > 0:
        request_node_ids = set(req.origin_node_id for req in all_requests)
        topology_leaf_ids = set(node.node_id for node in topology.leaf_nodes)
        
        print(f"\n  Request Node Validation:")
        print(f"    Unique node IDs in CSV:      {len(request_node_ids)}")
        print(f"    Topology leaf node IDs:      {len(topology_leaf_ids)}")
        
        matching = request_node_ids & topology_leaf_ids
        missing_in_topology = request_node_ids - topology_leaf_ids
        unused_topology_nodes = topology_leaf_ids - request_node_ids
        
        print(f"    Matching node IDs:           {len(matching)} {'✓' if len(matching) == len(topology_leaf_ids) else '✗'}")
        if matching:
            print(f"      {', '.join(sorted(matching))}")
        
        if missing_in_topology:
            print(f"    ⚠ CSV has {len(missing_in_topology)} node IDs not in topology:")
            print(f"      {', '.join(sorted(missing_in_topology))}")
            print(f"      These requests were SKIPPED during load!")
        
        if unused_topology_nodes:
            print(f"    ⚠ Topology has {len(unused_topology_nodes)} leaf nodes with NO requests:")
            print(f"      {', '.join(sorted(unused_topology_nodes))}")
    print()

    # Clamp steps to the available request count
    steps = sorted(set(min(n, n_total) for n in steps))
    print(f"  Effective steps: {steps}\n")

    # ?? Build DDQN orchestrator (inference only, epsilon=0) ???????????????
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

    # Policy registry
    # make_ddqn_policy wraps the agent so select_action() feeds the Q-network
    ddqn_action_fn : ActionFn = make_ddqn_policy(orchestrator.agent)

    # Run all (policy x load-level) combinations
    results   : Dict[str, List[dict]] = {p: [] for p in _POLICIES}
    all_rows  : List[dict]             = []

    for policy_name in _POLICIES:
        print(f"  Running: {policy_name}")
        print(_DIV)

        for n in steps:
            subset = all_requests[:n]

            if policy_name == "DDQN":
                m = _run_ddqn_policy(
                    requests      = subset,
                    topology      = topology,
                    catalogue     = catalogue,
                    orchestrator  = orchestrator,
                    prov_interval = prov_interval,
                )
            else:
                raise ValueError(f"Unknown policy: {policy_name}")

            results[policy_name].append(m)
            all_rows.append(m)

            print(
                f"    N={n:>6}  "
                f"Hit={m['hit_rate_mean']*100:>5.1f}%  "
                f"Local={m['local_rate']*100:>5.1f}%  "
                f"Cloud={m['cloud_rate']*100:>5.1f}%  "
                f"Drop={m['drop_rate']*100:>4.1f}%  "
                f"Rev={m['total_revenue']:>9.2f}  "
                f"Exec={m['executed']:>6}  Drop={m['dropped']:>5}"
            )
        print()

    # ?? Write CSV ??????????????????????????????????????????????????????????
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Results CSV saved -> {out_csv}")
    # Per-policy breakdown table
    _print_per_policy_table(results, steps)

    # Delta summary at max load
    _print_summary_table(results, steps)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="compare_policies.py",
        description="Compare DDQN vs Greedy vs Static Baseline cache provisioning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python compare_policies.py\n\n"
            "  python compare_policies.py \\\n"
            "      --checkpoint checkpoints/ddqn_finetune_v4.pt\n\n"
            "  python compare_policies.py \\\n"
            "      --checkpoint checkpoints/ddqn_finetuned.pt \\\n"
            "      --steps 100 500 1000 5000 10000 \\\n"
            "      --out-csv results/policy_comparison.csv \\\n"
            "      --out-plot results/policy_comparison.png\n"
        ),
    )
    p.add_argument(
        "--checkpoint", type=str, default=_DEFAULT_CHECKPOINT,
        help=f"DDQN checkpoint .pt file (default: {_DEFAULT_CHECKPOINT})",
    )
    p.add_argument(
        "--requests", type=str, default=_DEFAULT_REQUESTS,
        help=f"Request trace CSV (default: {_DEFAULT_REQUESTS})",
    )
    p.add_argument(
        "--services", type=str, default=_DEFAULT_SERVICES,
        help=f"Service catalogue CSV (default: {_DEFAULT_SERVICES})",
    )
    p.add_argument(
        "--steps", type=int, nargs="+", default=_DEFAULT_STEPS,
        help="Request load levels to evaluate (default: 100 250 500 1000 2000 5000 10000)",
    )
    p.add_argument(
        "--prov", type=int, default=PROVISIONING_INTERVAL,
        help=f"Provisioning interval in requests (default: {PROVISIONING_INTERVAL})",
    )
    p.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})",
    )
    p.add_argument(
        "--metric", type=str, 
        choices=["local", "cloud", "drop", "revenue", "all"],
        default="all",
        help="Which metric to plot: local (execution rate), cloud (escalation rate), drop (rate), revenue, or all (default: all)",
    )
    p.add_argument(
        "--out-csv", type=str, default=_DEFAULT_OUT_CSV,
        help=f"Output CSV path (default: {_DEFAULT_OUT_CSV})",
    )
    p.add_argument(
        "--out-plot", type=str, default=_DEFAULT_OUT_PLOT,
        help=f"Output PNG path (default: {_DEFAULT_OUT_PLOT})",
    )
    p.add_argument("" \
        "--number-levels" ,type=int, default=_DEFAULT_NUM_LEVELS,
        help=f"Output tree levels (default:{_DEFAULT_NUM_LEVELS})",
    )
    p.add_argument("" \
        "--number-children" ,type=int, default=_DEFAULT_NUM_CHILDREN,
        help=f"Output tree children (default:{_DEFAULT_NUM_CHILDREN})",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        checkpoint    = args.checkpoint,
        requests_csv  = args.requests,
        services_csv  = args.services,
        steps         = sorted(set(args.steps)),
        prov_interval = args.prov,
        seed          = args.seed,
        metric        = args.metric,
        out_csv       = args.out_csv,
        out_plot      = args.out_plot,
        number_levels = args.number_levels,
        number_children= args.number_children
    )
