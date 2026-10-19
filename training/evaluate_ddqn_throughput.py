# -*- coding: utf-8 -*-
"""
evaluate_ddqn_throughput.py
---------------------------
Evaluates the trained DDQN agent across increasing request loads
taken directly from a sample_requests.csv file. No retraining occurs.

The checkpoint is loaded once with epsilon=0 (greedy policy).
For each load level N (first N rows of the CSV) the script runs the
DDQN provisioning agent and collects:

  1. Hit rate mean + std-dev  -- mean and stability of rolling hit rate
  2. Local execution rate     -- requests served at origin leaf / total
  3. Cloud escalation rate    -- requests escalated to cloud / total
  4. Drop rate                -- deadline violations / total
  5. Total revenue            -- sum(revenue x omega) for executed requests
  6. Revenue by criticality tier
  7. Revenue per request      -- economic quality per served request

All results are written to a single CSV file.

Usage
-----
  # Use default paths
  python evaluate_ddqn_throughput.py

  # Specify request trace explicitly
  python evaluate_ddqn_throughput.py --requests training/data/sample_requests.csv

  # Full control
  python evaluate_ddqn_throughput.py
      --requests   training/data/sample_requests.csv
      --services   training/data/sample_services.csv
      --checkpoint checkpoints/ddqn_finetuned.pt
      --steps      100 500 1000 2000 5000
      --out        results/ddqn_throughput_metrics.csv

Output CSV columns
------------------
  n_requests, hit_rate_mean, hit_rate_std,
  local_rate, cloud_rate, drop_rate,
  total_revenue,
  rev_mission_critical, rev_high, rev_standard, rev_best_effort,
  rev_per_request, executed, dropped
"""

import argparse
import csv
import os
import sys
import random
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from edge_caching.config import (
    NUM_LEVELS, NUM_CHILDREN, NODE_CACHE_CAP,
    PROVISIONING_INTERVAL, RANDOM_SEED, RESULTS_DIR,
)
from edge_caching.core.service import Service, generate_service_catalogue
from edge_caching.core.request import Request
from edge_caching.network.topology import NetworkTopology
from edge_caching.control.orchestrator import EdgeOrchestrator
from edge_caching.utils.simulation.request_generator import load_requests_from_csv

_DEFAULT_REQUESTS   = os.path.join(_ROOT, "training", "data", "sample_requests.csv")
_DEFAULT_SERVICES   = os.path.join(_ROOT, "training", "data", "sample_services.csv")
_DEFAULT_CHECKPOINT = os.path.join(_ROOT, "checkpoints", "ddqn_finetuned.pt")
_DEFAULT_OUT_CSV    = os.path.join(_ROOT, RESULTS_DIR, "ddqn_throughput_metrics.csv")
_DEFAULT_STEPS      = [100, 250, 500, 1000, 2000, 5000,6000,7000,8000,10000]

_TIERS = ["mission_critical", "high", "standard", "best_effort"]

_SEP = "=" * 76
_DIV = "-" * 76

_CSV_COLUMNS = [
    "n_requests",
    "hit_rate_mean", "hit_rate_std",
    "local_rate", "cloud_rate", "drop_rate",
    "total_revenue",
    "rev_mission_critical", "rev_high", "rev_standard", "rev_best_effort",
    "rev_per_request",
    "executed", "dropped",
]


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
# World reset between load runs
# ---------------------------------------------------------------------------

def _reset_world(topology: NetworkTopology, catalogue: Dict[int, Service]) -> None:
    """Clear all caches, queues, counters then re-warm from priority_init."""
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
# Execution tier classifier
# ---------------------------------------------------------------------------

def _classify_tier(req: Request) -> str:
    """Return which tier handled this request: leaf / peer / parent / cloud / dropped."""
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
# Single load-level evaluation
# ---------------------------------------------------------------------------

def _evaluate_load(
    requests      : List[Request],
    topology      : NetworkTopology,
    catalogue     : Dict[int, Service],
    orchestrator  : EdgeOrchestrator,
    prov_interval : int,
) -> dict:
    """
    Run the DDQN agent on a fixed request list and collect all metrics.

    Hit rate stability is computed as the std-dev of a rolling hit rate
    series (window = 50 requests). A lower std-dev means the cache policy
    is consistently effective rather than spiking up and down.
    """
    _reset_world(topology, catalogue)

    tiers             = defaultdict(int)
    revenue_total     = 0.0
    revenue_by_tier   = {t: 0.0 for t in _TIERS}
    prov_calls        = 0
    total             = len(requests)
    total_prov        = max(total // prov_interval, 1)
    executed_flags: List[int] = []

    for idx, req in enumerate(requests):
        topology.node_map[req.origin_node_id].receive_request(req, catalogue)

        tier = _classify_tier(req)
        tiers[tier] += 1
        executed_flags.append(1 if req.status == "executed" else 0)

        if req.status == "executed":
            svc = catalogue[req.service_id]
            rv  = svc.revenue * svc.omega
            revenue_total += rv
            revenue_by_tier[svc.criticality] += rv

        if (idx + 1) % prov_interval == 0:
            prov_calls += 1
            orchestrator.provision(
                episode_done = (prov_calls == total_prov),
                sim_time_ms  = req.arrival_time,
            )

    executed = sum(executed_flags)
    dropped  = tiers["dropped"]

    # Rolling hit rate -- window=50 requests
    window    = 50
    flags_arr = np.array(executed_flags, dtype=float)
    if len(flags_arr) >= window:
        rolling_hr = np.convolve(
            flags_arr,
            np.ones(window) / window,
            mode="valid",
        )
        hr_mean = float(rolling_hr.mean())
        hr_std  = float(rolling_hr.std())
    else:
        hr_mean = float(flags_arr.mean()) if len(flags_arr) > 0 else 0.0
        hr_std  = 0.0

    orchestrator.end_episode(
        episode_reward   = 0.0,
        total_executed   = executed,
        total_requests   = total,
        total_forwarded  = tiers["peer"] + tiers["parent"] + tiers["cloud"],
        total_violations = dropped,
        avg_loss         = 0.0,
    )

    return {
        "n_requests"            : total,
        "hit_rate_mean"         : round(hr_mean,                                   6),
        "hit_rate_std"          : round(hr_std,                                    6),
        "local_rate"            : round(tiers["leaf"]  / max(total, 1),            6),
        "cloud_rate"            : round(tiers["cloud"] / max(total, 1),            6),
        "drop_rate"             : round(dropped        / max(total, 1),            6),
        "total_revenue"         : round(revenue_total,                             4),
        "rev_mission_critical"  : round(revenue_by_tier["mission_critical"],       4),
        "rev_high"              : round(revenue_by_tier["high"],                   4),
        "rev_standard"          : round(revenue_by_tier["standard"],               4),
        "rev_best_effort"       : round(revenue_by_tier["best_effort"],            4),
        "rev_per_request"       : round(revenue_total / max(executed, 1),          6),
        "executed"              : executed,
        "dropped"               : dropped,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run(
    requests_csv  : str           = _DEFAULT_REQUESTS,
    services_csv  : str           = _DEFAULT_SERVICES,
    checkpoint    : Optional[str] = _DEFAULT_CHECKPOINT,
    steps         : List[int]     = _DEFAULT_STEPS,
    prov_interval : int           = PROVISIONING_INTERVAL,
    seed          : int           = RANDOM_SEED,
    out_csv       : str           = _DEFAULT_OUT_CSV,
) -> None:

    random.seed(seed)
    np.random.seed(seed)

    print(_SEP)
    print("  DDQN Throughput Evaluation  --  Load Scaling")
    print(f"  Requests file : {requests_csv}")
    print(f"  Services file : {services_csv}")
    print(f"  Checkpoint    : {checkpoint}")
    print(f"  Load steps    : {steps}")
    print(f"  Prov interval : {prov_interval}")
    print(f"  Output CSV    : {out_csv}")
    print(_SEP)

    # -- Service catalogue ------------------------------------------------
    if services_csv and os.path.exists(services_csv):
        catalogue = _load_services(services_csv)
        print(f"  Loaded {len(catalogue)} services from {services_csv}")
    else:
        catalogue = {s.service_id: s
                     for s in generate_service_catalogue(50, seed=seed)}
        print(f"  Generated {len(catalogue)} random services")

    # -- Topology ---------------------------------------------------------
    topology = NetworkTopology(
        num_levels   = NUM_LEVELS,
        num_children = NUM_CHILDREN,
        cache_cap    = NODE_CACHE_CAP,
    )
    topology.prime_cloud_cache(catalogue)
    topology.priority_init(catalogue)

    # -- Load request trace -----------------------------------------------
    if not os.path.exists(requests_csv):
        print(f"\n  ERROR: requests CSV not found: {requests_csv}")
        print("  Generate one with:")
        print("    python training/data/generate_requests.py")
        sys.exit(1)

    all_requests = load_requests_from_csv(
        requests_csv, catalogue, topology.node_map,
    )
    n_total = len(all_requests)
    print(f"  Loaded {n_total} requests from CSV")

    # Clamp steps to available request count
    steps = sorted(set(min(n, n_total) for n in steps))
    print(f"  Effective steps: {steps}")
    print()

    # -- DDQN orchestrator (greedy) ---------------------------------------
    orchestrator = EdgeOrchestrator(nodes=topology.all_nodes, catalogue=catalogue)
    if checkpoint and os.path.exists(checkpoint):
        orchestrator.load_checkpoint(checkpoint)
        orchestrator.agent.epsilon = 0.0
        print(f"  Checkpoint loaded  ->  epsilon=0 (greedy policy)")
    else:
        orchestrator.agent.epsilon = 0.0
        print("  No checkpoint found  ->  untrained weights, epsilon=0")
    print()

    # -- Table header -----------------------------------------------------
    print(
        f"  {'N':>6}  "
        f"{'HitMean%':>9}  {'HitStd%':>8}  "
        f"{'Local%':>7}  {'Cloud%':>7}  {'Drop%':>6}  "
        f"{'Revenue':>10}  "
        f"{'MC_Rev':>8}  {'Hi_Rev':>8}  {'Std_Rev':>8}  {'BE_Rev':>8}"
    )
    print(_DIV)

    results: List[dict] = []

    for n in steps:
        m = _evaluate_load(
            requests      = all_requests[:n],
            topology      = topology,
            catalogue     = catalogue,
            orchestrator  = orchestrator,
            prov_interval = prov_interval,
        )
        results.append(m)

        print(
            f"  {n:>6}  "
            f"{m['hit_rate_mean']*100:>8.2f}%  "
            f"{m['hit_rate_std']*100:>7.2f}%  "
            f"{m['local_rate']*100:>6.2f}%  "
            f"{m['cloud_rate']*100:>6.2f}%  "
            f"{m['drop_rate']*100:>5.2f}%  "
            f"{m['total_revenue']:>10.2f}  "
            f"{m['rev_mission_critical']:>8.2f}  "
            f"{m['rev_high']:>8.2f}  "
            f"{m['rev_standard']:>8.2f}  "
            f"{m['rev_best_effort']:>8.2f}"
        )

    # -- Write CSV --------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(results)

    print(_DIV)
    print(f"\n  Results saved -> {out_csv}")

    # -- Console summary --------------------------------------------------
    print()
    print("  SUMMARY  (first load level vs last load level)")
    print(_DIV)
    first, last = results[0], results[-1]

    summary_rows = [
        # (label,                         column,                  is_percent)
        ("Hit rate mean",                 "hit_rate_mean",         True),
        ("Hit rate std (stability)",      "hit_rate_std",          True),
        ("Local execution rate",          "local_rate",            True),
        ("Cloud escalation rate",         "cloud_rate",            True),
        ("Drop rate",                     "drop_rate",             True),
        ("Total revenue",                 "total_revenue",         False),
        ("Revenue per request",           "rev_per_request",       False),
        ("Revenue -- mission_critical",   "rev_mission_critical",  False),
        ("Revenue -- high",               "rev_high",              False),
        ("Revenue -- standard",           "rev_standard",          False),
        ("Revenue -- best_effort",        "rev_best_effort",       False),
    ]

    for label, col, pct in summary_rows:
        fv = first[col]
        lv = last[col]
        d  = lv - fv
        if pct:
            print(
                f"  {label:<34}  "
                f"N={steps[0]:>5}: {fv*100:>6.2f}%   "
                f"N={steps[-1]:>5}: {lv*100:>6.2f}%   "
                f"delta: {d*100:>+6.2f} pp"
            )
        else:
            print(
                f"  {label:<34}  "
                f"N={steps[0]:>5}: {fv:>9.3f}   "
                f"N={steps[-1]:>5}: {lv:>9.3f}   "
                f"delta: {d:>+9.3f}"
            )

    print(_SEP)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="evaluate_ddqn_throughput.py",
        description=(
            "Evaluate DDQN provisioning agent performance across "
            "increasing request loads from a sample_requests.csv file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluate_ddqn_throughput.py\n\n"
            "  python evaluate_ddqn_throughput.py \\\n"
            "      --requests training/data/sample_requests.csv\n\n"
            "  python evaluate_ddqn_throughput.py \\\n"
            "      --requests   training/data/sample_requests.csv \\\n"
            "      --services   training/data/sample_services.csv \\\n"
            "      --checkpoint checkpoints/ddqn_finetuned.pt \\\n"
            "      --steps      100 500 1000 2500 5000 \\\n"
            "      --out        results/ddqn_throughput_metrics.csv\n"
        ),
    )
    p.add_argument(
        "--requests", type=str, default=_DEFAULT_REQUESTS,
        help=f"Path to request trace CSV (default: {_DEFAULT_REQUESTS})",
    )
    p.add_argument(
        "--services", type=str, default=_DEFAULT_SERVICES,
        help=f"Path to service catalogue CSV (default: {_DEFAULT_SERVICES})",
    )
    p.add_argument(
        "--checkpoint", type=str, default=_DEFAULT_CHECKPOINT,
        help=f"Trained DDQN checkpoint .pt file (default: {_DEFAULT_CHECKPOINT})",
    )
    p.add_argument(
        "--steps", type=int, nargs="+", default=_DEFAULT_STEPS,
        help="Request load levels to evaluate (default: 100 250 500 1000 2000 5000)",
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
        "--out", type=str, default=_DEFAULT_OUT_CSV,
        help=f"Output CSV path (default: {_DEFAULT_OUT_CSV})",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        requests_csv  = args.requests,
        services_csv  = args.services,
        checkpoint    = args.checkpoint,
        steps         = sorted(set(args.steps)),
        prov_interval = args.prov,
        seed          = args.seed,
        out_csv       = args.out,
    )
