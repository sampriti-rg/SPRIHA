"""
evaluate_custom_requests.py
---------------------------
Run a custom request trace (CSV) through two back-to-back passes:

  Pass A  -- BASELINE  : static priority_init cache, NO DDQN provisioning.
  Pass B  -- DDQN      : same static warm-start, then DDQN agent provisions
             every PROVISIONING_INTERVAL requests (greedy epsilon=0 if a
             checkpoint is supplied, else training epsilon schedule).

Both passes use IDENTICAL requests loaded from the CSV so results are
directly comparable. The improvement delta is printed to the console and
written to results/custom_eval_report.txt + results/custom_eval_requests.csv.

Metrics collected per pass
--------------------------
  hit_rate         executed / total
  drop_rate        dropped  / total
  leaf_pct         executed at origin leaf / total
  peer_pct         executed at peer        / total
  parent_pct       executed at parent edge / total
  cloud_pct        escalated to cloud      / total
  avg_latency_ms   mean latency consumed by executed requests
  total_revenue    sum of (revenue * omega) for executed requests
  sla_violations   total deadline-expired requests

Usage
-----
  # Run with auto-generated synthetic traffic (default)
  python training/evaluate_custom_requests.py

  # Use your own request CSV
  python training/evaluate_custom_requests.py --requests training/data/sample_requests.csv

  # Use a trained checkpoint for the DDQN pass
  python training/evaluate_custom_requests.py \\
      --requests training/data/sample_requests.csv \\
      --checkpoint checkpoints/ddqn_final.pt

  # Use the sample service catalogue instead of random services
  python training/evaluate_custom_requests.py \\
      --requests training/data/sample_requests.csv \\
      --services  training/data/sample_services.csv \\
      --checkpoint checkpoints/ddqn_final.pt
"""

import argparse
import copy
import csv
import datetime
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge_caching.config import (
    NUM_LEVELS, NUM_CHILDREN, NODE_CACHE_CAP, NUM_SERVICES,
    PROVISIONING_INTERVAL, RANDOM_SEED, RESULTS_DIR, TRAFFIC_MODEL,
)
from edge_caching.core.service import Service, generate_service_catalogue
from edge_caching.core.request import Request, _request_counter
from edge_caching.network.topology import NetworkTopology
from edge_caching.control.orchestrator import EdgeOrchestrator
from edge_caching.utils.simulation.request_generator import load_requests_from_csv

_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR    = os.path.join(_ROOT, "training", "data")
_DEFAULT_CSV = os.path.join(_DATA_DIR, "sample_requests.csv")
_DEFAULT_SVC = os.path.join(_DATA_DIR, "sample_services.csv")
_SEP = "=" * 80
_DIV = "-" * 80

RESULT_CSV    = "custom_eval_requests.csv"
RESULT_REPORT = "custom_eval_report.txt"

REQUEST_COLUMNS = [
    "pass", "request_id", "service_id", "criticality", "omega",
    "origin_node_id", "deadline_ms", "arrival_time_ms",
    "executed_at_node", "remaining_deadline_ms", "latency_consumed_ms",
    "status", "execution_tier", "hop_count",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_services_csv(csv_path: str) -> Dict[int, Service]:
    """Load service catalogue from sample_services.csv."""
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


def _clone_requests(requests: List[Request]) -> List[Request]:
    """
    Return fresh Request objects with the same (service_id, origin_node_id,
    deadline_ms, arrival_time) so both passes start from identical inputs.
    request_id is re-assigned sequentially from the global counter.
    """
    cloned = []
    for r in requests:
        cloned.append(Request(
            service_id     = r.service_id,
            origin_node_id = r.origin_node_id,
            deadline_ms    = r.deadline_ms,
            arrival_time   = r.arrival_time,
        ))
    return cloned


def _reset_catalogue(catalogue: Dict[int, Service]) -> None:
    """Reset all runtime counters without changing catalogue attributes."""
    for svc in catalogue.values():
        svc.reset_interval_counters()


def _reset_topology(topology: NetworkTopology) -> None:
    """Clear all caches, queues, counters, and revenue on every node."""
    for node in [topology.cloud_root] + topology.all_nodes:
        node.cache.clear()
        node.cache_sizes.clear()
        node._queue.clear()
        node.total_revenue = 0.0
        node.reset_interval_stats()


def _tier(req: Request) -> str:
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


def _latency(req: Request) -> float:
    return req.deadline_ms - req.remaining_deadline


def _collect_metrics(requests: List[Request], catalogue: Dict[int, Service]) -> dict:
    total   = len(requests)
    exe     = [r for r in requests if r.status == "executed"]
    dropped = [r for r in requests if r.status == "dropped"]
    tiers   = defaultdict(int)
    for r in requests:
        tiers[_tier(r)] += 1

    revenue = sum(
        catalogue[r.service_id].revenue * catalogue[r.service_id].omega
        for r in exe
    )

    return {
        "total"        : total,
        "executed"     : len(exe),
        "dropped"      : len(dropped),
        "hit_rate"     : len(exe)     / max(total, 1),
        "drop_rate"    : len(dropped) / max(total, 1),
        "leaf_pct"     : tiers["leaf"]   / max(total, 1),
        "peer_pct"     : tiers["peer"]   / max(total, 1),
        "parent_pct"   : tiers["parent"] / max(total, 1),
        "cloud_pct"    : tiers["cloud"]  / max(total, 1),
        "avg_latency"  : sum(_latency(r) for r in exe) / max(len(exe), 1),
        "total_revenue": revenue,
        "tiers"        : dict(tiers),
    }


def _row(req: Request, catalogue: Dict, pass_label: str) -> dict:
    svc = catalogue[req.service_id]
    return {
        "pass"                 : pass_label,
        "request_id"           : req.request_id,
        "service_id"           : req.service_id,
        "criticality"          : svc.criticality,
        "omega"                : f"{svc.omega:.1f}",
        "origin_node_id"       : req.origin_node_id,
        "deadline_ms"          : f"{req.deadline_ms:.2f}",
        "arrival_time_ms"      : f"{req.arrival_time:.3f}",
        "executed_at_node"     : req.executed_at_node or "-",
        "remaining_deadline_ms": f"{req.remaining_deadline:.2f}",
        "latency_consumed_ms"  : f"{_latency(req):.2f}",
        "status"               : req.status,
        "execution_tier"       : _tier(req),
        "hop_count"            : len(req.hop_path),
    }


# ---------------------------------------------------------------------------
# Single-pass runner
# ---------------------------------------------------------------------------

def _run_pass(
    label       : str,
    requests    : List[Request],
    topology    : NetworkTopology,
    catalogue   : Dict[int, Service],
    orchestrator: EdgeOrchestrator,
    prov_interval: int,
    enable_ddqn : bool,
    sim_times   : List[float],
) -> Tuple[List[Request], dict]:
    """
    Route all requests through the topology, optionally provisioning with DDQN.

    Parameters
    ----------
    label        : 'BASELINE' or 'DDQN'
    enable_ddqn  : if False, provision() is never called (baseline)
    sim_times    : pre-computed simulated clock per provision call
                   (derived from request arrival times)
    """
    total_prov  = len(requests) // prov_interval
    prov_calls  = 0
    ep_executed = 0
    ep_dropped  = 0
    ep_fwd      = 0

    for idx, req in enumerate(requests):
        topology.node_map[req.origin_node_id].receive_request(req, catalogue)
        if req.status == "executed":
            ep_executed += 1
            if req.executed_at_node != req.origin_node_id:
                ep_fwd += 1
        elif req.status == "dropped":
            ep_dropped += 1

        if enable_ddqn and (idx + 1) % prov_interval == 0:
            prov_calls += 1
            is_last    = (prov_calls == total_prov)
            sim_t      = sim_times[prov_calls - 1] if prov_calls <= len(sim_times) else 0.0
            orchestrator.provision(episode_done=is_last, sim_time_ms=sim_t)

    metrics = _collect_metrics(requests, catalogue)
    orchestrator.end_episode(
        episode_reward   = 0.0,
        total_executed   = ep_executed,
        total_requests   = len(requests),
        total_forwarded  = ep_fwd,
        total_violations = ep_dropped,
        avg_loss         = 0.0,
    )
    return requests, metrics


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate(
    requests_csv     : str           = _DEFAULT_CSV,
    services_csv     : Optional[str] = None,
    checkpoint_path  : Optional[str] = None,
    prov_interval    : int           = PROVISIONING_INTERVAL,
    seed             : int           = RANDOM_SEED,
    save_outputs     : bool          = True,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(_SEP)
    print("  Edge Caching DDQN  --  Custom Request Trace Evaluation")
    print(f"  Requests CSV : {requests_csv}")
    print(f"  Services     : {services_csv or 'auto-generated'}")
    print(f"  Checkpoint   : {checkpoint_path or 'none (training epsilon schedule)'}")
    print(f"  Started      : {ts}")
    print(_DIV)

    # Build catalogue
    if services_csv and os.path.exists(services_csv):
        catalogue = _load_services_csv(services_csv)
        print(f"  Loaded {len(catalogue)} services from {services_csv}")
    else:
        catalogue = {s.service_id: s for s in
                     generate_service_catalogue(NUM_SERVICES, seed=seed)}
        print(f"  Generated {len(catalogue)} services (random catalogue)")

    # Build topology
    topology = NetworkTopology(
        num_levels=NUM_LEVELS, num_children=NUM_CHILDREN, cache_cap=NODE_CACHE_CAP,
    )
    topology.prime_cloud_cache(catalogue)
    topology.priority_init(catalogue)

    # Load requests from CSV
    requests_master = load_requests_from_csv(
        requests_csv, catalogue, topology.node_map,
    )
    n = len(requests_master)
    print(f"  Loaded {n} requests from {requests_csv}")
    print()

    if n == 0:
        print("  ERROR: No valid requests loaded.  Check service_id / origin_node_id columns.")
        return

    # Pre-compute simulated arrival times for each provision call
    # (used so the DDQN placement cost knows the real elapsed interval)
    sim_times_ddqn: List[float] = []
    for i in range(prov_interval - 1, n, prov_interval):
        sim_times_ddqn.append(requests_master[i].arrival_time)

    # ----------------------------------------------------------------
    # PASS A -- BASELINE (no DDQN provisioning)
    # ----------------------------------------------------------------
    print("  [Pass A]  BASELINE  -- static priority_init cache, no provisioning ...")
    _reset_topology(topology)
    _reset_catalogue(catalogue)
    topology.prime_cloud_cache(catalogue)
    topology.priority_init(catalogue)

    orch_baseline = EdgeOrchestrator(nodes=topology.all_nodes, catalogue=catalogue)
    orch_baseline.agent.epsilon = 1.0   # not used (ddqn disabled), but set explicitly

    reqs_A = _clone_requests(requests_master)
    reqs_A, metrics_A = _run_pass(
        label        = "BASELINE",
        requests     = reqs_A,
        topology     = topology,
        catalogue    = catalogue,
        orchestrator = orch_baseline,
        prov_interval= prov_interval,
        enable_ddqn  = False,
        sim_times    = sim_times_ddqn,
    )
    print(f"         hit_rate={metrics_A['hit_rate']*100:.1f}%  "
          f"drop_rate={metrics_A['drop_rate']*100:.1f}%  "
          f"revenue={metrics_A['total_revenue']:.2f}")

    # ----------------------------------------------------------------
    # PASS B -- DDQN agent
    # ----------------------------------------------------------------
    print("  [Pass B]  DDQN     -- same warm-start, DDQN agent provisions every "
          f"{prov_interval} requests ...")
    _reset_topology(topology)
    _reset_catalogue(catalogue)
    topology.prime_cloud_cache(catalogue)
    topology.priority_init(catalogue)

    orch_ddqn = EdgeOrchestrator(nodes=topology.all_nodes, catalogue=catalogue)
    if checkpoint_path and os.path.exists(checkpoint_path):
        orch_ddqn.load_checkpoint(checkpoint_path)
        orch_ddqn.agent.epsilon = 0.0
        print(f"         Loaded checkpoint: {checkpoint_path}  (epsilon=0, greedy)")
    else:
        print("         No checkpoint -- using training epsilon schedule")

    reqs_B = _clone_requests(requests_master)
    reqs_B, metrics_B = _run_pass(
        label        = "DDQN",
        requests     = reqs_B,
        topology     = topology,
        catalogue    = catalogue,
        orchestrator = orch_ddqn,
        prov_interval= prov_interval,
        enable_ddqn  = True,
        sim_times    = sim_times_ddqn,
    )
    print(f"         hit_rate={metrics_B['hit_rate']*100:.1f}%  "
          f"drop_rate={metrics_B['drop_rate']*100:.1f}%  "
          f"revenue={metrics_B['total_revenue']:.2f}")

    # ----------------------------------------------------------------
    # Delta (improvement)
    # ----------------------------------------------------------------
    delta = {
        "hit_rate_pp"    : (metrics_B["hit_rate"]     - metrics_A["hit_rate"])     * 100,
        "drop_rate_pp"   : (metrics_B["drop_rate"]    - metrics_A["drop_rate"])    * 100,
        "leaf_pct_pp"    : (metrics_B["leaf_pct"]     - metrics_A["leaf_pct"])     * 100,
        "cloud_pct_pp"   : (metrics_B["cloud_pct"]    - metrics_A["cloud_pct"])    * 100,
        "latency_ms"     :  metrics_B["avg_latency"]  - metrics_A["avg_latency"],
        "revenue_abs"    :  metrics_B["total_revenue"] - metrics_A["total_revenue"],
        "revenue_pct"    : ((metrics_B["total_revenue"] - metrics_A["total_revenue"])
                            / max(metrics_A["total_revenue"], 1e-6)) * 100,
        "violations_abs" :  metrics_B["dropped"]       - metrics_A["dropped"],
    }

    _print_comparison(metrics_A, metrics_B, delta, n, prov_interval, checkpoint_path)

    # ----------------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------------
    if save_outputs:
        results_dir  = os.path.join(_ROOT, RESULTS_DIR)
        os.makedirs(results_dir, exist_ok=True)
        csv_path     = os.path.join(results_dir, RESULT_CSV)
        report_path  = os.path.join(results_dir, RESULT_REPORT)

        all_rows = (
            [_row(r, catalogue, "BASELINE") for r in reqs_A] +
            [_row(r, catalogue, "DDQN")     for r in reqs_B]
        )
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REQUEST_COLUMNS)
            writer.writeheader()
            writer.writerows(all_rows)

        _write_report(
            report_path, ts, requests_csv, services_csv, checkpoint_path,
            n, prov_interval, metrics_A, metrics_B, delta, topology, catalogue,
        )
        print(f"\n  Per-request CSV  -> {csv_path}")
        print(f"  Summary report   -> {report_path}")
    print(_SEP)


# ---------------------------------------------------------------------------
# Console comparison table
# ---------------------------------------------------------------------------

def _print_comparison(mA, mB, delta, n_req, prov_interval, checkpoint):
    print()
    print(_SEP)
    print("  EVALUATION RESULTS  --  BASELINE  vs  DDQN")
    print(_DIV)
    print(f"  {'Metric':<30} {'BASELINE':>12}  {'DDQN':>12}  {'Delta':>12}")
    print(_DIV)

    rows = [
        ("Total requests",     f"{n_req}",                      f"{n_req}",                      ""),
        ("Executed",           f"{mA['executed']}",              f"{mB['executed']}",
            f"{mB['executed']-mA['executed']:+d}"),
        ("Dropped",            f"{mA['dropped']}",               f"{mB['dropped']}",
            f"{mB['dropped']-mA['dropped']:+d}"),
        ("Hit rate",           f"{mA['hit_rate']*100:.2f}%",     f"{mB['hit_rate']*100:.2f}%",
            f"{delta['hit_rate_pp']:+.2f} pp"),
        ("Drop rate",          f"{mA['drop_rate']*100:.2f}%",    f"{mB['drop_rate']*100:.2f}%",
            f"{delta['drop_rate_pp']:+.2f} pp"),
        ("Leaf hit %",         f"{mA['leaf_pct']*100:.2f}%",     f"{mB['leaf_pct']*100:.2f}%",
            f"{delta['leaf_pct_pp']:+.2f} pp"),
        ("Cloud escalation %", f"{mA['cloud_pct']*100:.2f}%",    f"{mB['cloud_pct']*100:.2f}%",
            f"{delta['cloud_pct_pp']:+.2f} pp"),
        ("Avg latency (ms)",   f"{mA['avg_latency']:.2f}",       f"{mB['avg_latency']:.2f}",
            f"{delta['latency_ms']:+.2f} ms"),
        ("Total revenue",      f"{mA['total_revenue']:.3f}",     f"{mB['total_revenue']:.3f}",
            f"{delta['revenue_abs']:+.3f} ({delta['revenue_pct']:+.1f}%)"),
    ]
    for label, va, vb, dv in rows:
        print(f"  {label:<30} {va:>12}  {vb:>12}  {dv:>12}")

    print(_DIV)
    print("\n  Tier breakdown:")
    print(f"  {'Tier':<16} {'BASELINE':>10}  {'DDQN':>10}  {'Delta':>10}")
    print("  " + "-" * 48)
    for tier in ["leaf", "peer", "parent", "cloud", "dropped"]:
        ca = mA["tiers"].get(tier, 0)
        cb = mB["tiers"].get(tier, 0)
        print(f"  {tier:<16} {ca:>10}  {cb:>10}  {cb-ca:>+10}")
    print(_SEP)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_report(path, ts, req_csv, svc_csv, ckpt, n_req, prov_interval,
                  mA, mB, delta, topology, catalogue):
    with open(path, "w", encoding="utf-8") as f:
        f.write(_SEP + "\n")
        f.write("  EDGE CACHING DDQN  --  CUSTOM REQUEST EVALUATION REPORT\n")
        f.write(f"  Timestamp        : {ts}\n")
        f.write(f"  Requests CSV     : {req_csv}\n")
        f.write(f"  Services CSV     : {svc_csv or 'auto-generated'}\n")
        f.write(f"  Checkpoint       : {ckpt or 'none'}\n")
        f.write(_SEP + "\n\n")

        f.write("[1] RUN PARAMETERS\n" + _DIV + "\n")
        f.write(f"  Topology         : {NUM_LEVELS} levels, {NUM_CHILDREN} children, "
                f"cache_cap={NODE_CACHE_CAP}\n")
        f.write(f"  Requests         : {n_req}\n")
        f.write(f"  Prov interval    : {prov_interval}\n")
        f.write(f"  Prov calls (DDQN): {n_req // prov_interval}\n")
        f.write(_DIV + "\n\n")

        f.write("[2] METRIC COMPARISON\n" + _DIV + "\n")
        f.write(f"  {'Metric':<30} {'BASELINE':>12}  {'DDQN':>12}  {'Delta':>14}\n")
        f.write(_DIV + "\n")
        rows = [
            ("Total requests",     n_req,   n_req,   ""),
            ("Executed",           mA["executed"],       mB["executed"],
                f"{mB['executed']-mA['executed']:+d}"),
            ("Dropped",            mA["dropped"],        mB["dropped"],
                f"{mB['dropped']-mA['dropped']:+d}"),
            ("Hit rate",           f"{mA['hit_rate']*100:.2f}%",
                f"{mB['hit_rate']*100:.2f}%",
                f"{delta['hit_rate_pp']:+.2f} pp"),
            ("Drop rate",          f"{mA['drop_rate']*100:.2f}%",
                f"{mB['drop_rate']*100:.2f}%",
                f"{delta['drop_rate_pp']:+.2f} pp"),
            ("Leaf hit %",         f"{mA['leaf_pct']*100:.2f}%",
                f"{mB['leaf_pct']*100:.2f}%",
                f"{delta['leaf_pct_pp']:+.2f} pp"),
            ("Cloud escalation %", f"{mA['cloud_pct']*100:.2f}%",
                f"{mB['cloud_pct']*100:.2f}%",
                f"{delta['cloud_pct_pp']:+.2f} pp"),
            ("Avg latency (ms)",   f"{mA['avg_latency']:.2f}",
                f"{mB['avg_latency']:.2f}",
                f"{delta['latency_ms']:+.2f} ms"),
            ("Total revenue",      f"{mA['total_revenue']:.3f}",
                f"{mB['total_revenue']:.3f}",
                f"{delta['revenue_abs']:+.3f} ({delta['revenue_pct']:+.1f}%)"),
        ]
        for label, va, vb, dv in rows:
            f.write(f"  {label:<30} {str(va):>12}  {str(vb):>12}  {str(dv):>14}\n")
        f.write(_DIV + "\n\n")

        f.write("[3] TIER BREAKDOWN\n" + _DIV + "\n")
        f.write(f"  {'Tier':<16} {'BASELINE':>10}  {'DDQN':>10}  {'Delta':>10}\n")
        f.write(_DIV + "\n")
        for tier in ["leaf", "peer", "parent", "cloud", "dropped"]:
            ca = mA["tiers"].get(tier, 0)
            cb = mB["tiers"].get(tier, 0)
            f.write(f"  {tier:<16} {ca:>10}  {cb:>10}  {cb-ca:>+10}\n")
        f.write(_DIV + "\n\n")

        f.write("[4] FINAL CACHE STATE  (after DDQN pass)\n" + _DIV + "\n")
        f.write(f"  {'Node':<14} {'Lvl':>4} {'Cap':>5} {'Used%':>7}  Cached Services\n")
        f.write(_DIV + "\n")
        for node in [topology.cloud_root] + topology.all_nodes:
            f.write(f"  {node.node_id:<14} {node.level:>4} {node.cache_cap:>5} "
                    f"{node.cache_utilization()*100:>6.1f}%  {sorted(node.cache)}\n")
        f.write(_DIV + "\n\n")

        f.write("[5] SERVICE CATALOGUE\n" + _DIV + "\n")
        f.write(f"  {'ID':<5} {'Name / Criticality':<28} {'Omega':>6} "
                f"{'Pop':>8} {'Rev':>6} {'InitCyc':>12}\n")
        f.write(_DIV + "\n")
        for sid, svc in sorted(catalogue.items()):
            f.write(f"  {sid:<5} {svc.criticality:<28} {svc.omega:>6.1f} "
                    f"{svc.popularity:>8.4f} {svc.revenue:>6.2f} "
                    f"{svc.init_cpu_cycles:>12,.0f}\n")
        f.write(_DIV + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate DDQN improvement on a custom request trace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python training/evaluate_custom_requests.py\n"
            "  python training/evaluate_custom_requests.py "
            "--requests training/data/sample_requests.csv\n"
            "  python training/evaluate_custom_requests.py "
            "--requests training/data/sample_requests.csv "
            "--services training/data/sample_services.csv "
            "--checkpoint checkpoints/ddqn_final.pt\n"
        ),
    )
    p.add_argument("--requests",    type=str, default=_DEFAULT_CSV,
                   help=f"Request trace CSV  (default: {_DEFAULT_CSV})")
    p.add_argument("--services",    type=str, default=None,
                   help=f"Service catalogue CSV  (default: auto-generated)")
    p.add_argument("--checkpoint",  type=str, default=None,
                   help="Trained DDQN checkpoint .pt file  (default: none)")
    p.add_argument("--prov",        type=int, default=PROVISIONING_INTERVAL,
                   help=f"Provisioning interval  (default: {PROVISIONING_INTERVAL})")
    p.add_argument("--seed",        type=int, default=RANDOM_SEED,
                   help=f"Random seed  (default: {RANDOM_SEED})")
    p.add_argument("--no-save",     action="store_true",
                   help="Skip writing CSV and report to results/")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        requests_csv    = args.requests,
        services_csv    = args.services,
        checkpoint_path = args.checkpoint,
        prov_interval   = args.prov,
        seed            = args.seed,
        save_outputs    = not args.no_save,
    )
