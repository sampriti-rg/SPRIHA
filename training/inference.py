"""
inference.py - DDQN edge-caching inference.
Loads a trained checkpoint (epsilon=0, greedy policy) and runs the simulation.
Usage: python inference.py [--checkpoint PATH] [--episodes N] [--requests N]
"""
import argparse
import csv
import datetime
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

# Ensure the project root is on sys.path when run as  python training/inference.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge_caching.config import (
    NUM_LEVELS, NUM_CHILDREN, NODE_CACHE_CAP, NUM_SERVICES,
    REQUESTS_PER_EPISODE, PROVISIONING_INTERVAL, RANDOM_SEED,
    RESULTS_DIR, TRAFFIC_MODEL,
)
from edge_caching.core.service import generate_service_catalogue
from edge_caching.core.request import Request
from edge_caching.network.topology import NetworkTopology
from edge_caching.utils.simulation.request_generator import RequestGenerator
from edge_caching.control.orchestrator import EdgeOrchestrator

_ROOT              = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR     = os.path.join(_ROOT, "checkpoints")
DEFAULT_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "ddqn_final.pt")
INFERENCE_REPORT   = "inference_report.txt"
INFERENCE_CSV      = "inference_requests.csv"
_SEP = "=" * 72
_DIV = "-" * 72
REQUEST_COLUMNS = [
    "episode", "request_id", "service_id", "criticality", "omega",
    "origin_node_id", "deadline_ms", "executed_at_node",
    "remaining_deadline_ms", "latency_consumed_ms",
    "status", "execution_tier", "hop_count",
]


def _execution_tier(req: Request) -> str:
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


def _row(req: Request, catalogue: Dict, episode: int) -> dict:
    svc = catalogue[req.service_id]
    return {
        "episode"               : episode,
        "request_id"            : req.request_id,
        "service_id"            : req.service_id,
        "criticality"           : svc.criticality,
        "omega"                 : f"{svc.omega:.1f}",
        "origin_node_id"        : req.origin_node_id,
        "deadline_ms"           : f"{req.deadline_ms:.2f}",
        "executed_at_node"      : req.executed_at_node or "-",
        "remaining_deadline_ms" : f"{req.remaining_deadline:.2f}",
        "latency_consumed_ms"   : f"{_latency(req):.2f}",
        "status"                : req.status,
        "execution_tier"        : _execution_tier(req),
        "hop_count"             : len(req.hop_path),
    }


def _stats(requests: List[Request]) -> dict:
    total = len(requests)
    exe   = [r for r in requests if r.status == "executed"]
    drp   = [r for r in requests if r.status == "dropped"]
    tiers = defaultdict(int)
    for r in requests:
        tiers[_execution_tier(r)] += 1
    return {
        "total"         : total,
        "executed"      : len(exe),
        "dropped"       : len(drp),
        "hit_rate"      : len(exe) / max(total, 1),
        "avg_latency_ms": sum(_latency(r) for r in exe) / max(len(exe), 1),
        "tiers"         : dict(tiers),
    }


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_trained_orchestrator(
    checkpoint_path : str,
    num_levels      : int = NUM_LEVELS,
    num_children    : int = NUM_CHILDREN,
    cache_cap       : int = NODE_CACHE_CAP,
    num_services    : int = NUM_SERVICES,
    seed            : int = RANDOM_SEED,
    services_csv    : str = None,
) -> Tuple[EdgeOrchestrator, NetworkTopology, Dict, RequestGenerator]:
    """
    Rebuild the world and load trained DDQN weights.
    Forces epsilon=0.0 for a fully greedy inference policy.

    Parameters
    ----------
    checkpoint_path : str
        Path to trained checkpoint
    num_levels, num_children, cache_cap : topology parameters
    num_services : int
        Number of services (used only if services_csv is None)
    seed : int
        Random seed
    services_csv : str, optional
        Path to custom service catalogue CSV. If provided, loads from file
        instead of generating random catalogue.

    Returns
    -------
    (orchestrator, topology, catalogue, req_gen)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run  python train.py  first to generate a checkpoint."
        )
    _seed_all(seed)
    
    # Load services from CSV or generate random
    if services_csv and os.path.exists(services_csv):
        print(f"  Loading services from: {services_csv}")
        from training.collect_stats import load_sample_services_with_names
        catalogue, names = load_sample_services_with_names(services_csv)
        print(f"  Loaded {len(catalogue)} services from CSV")
    else:
        cat_list  = generate_service_catalogue(num_services, seed=seed)
        catalogue = {s.service_id: s for s in cat_list}
    
    topology  = NetworkTopology(
        num_levels=num_levels, num_children=num_children, cache_cap=cache_cap,
    )
    topology.prime_cloud_cache(catalogue)
    topology.priority_init(catalogue)
    req_gen      = RequestGenerator(
        catalogue=catalogue, leaf_nodes=topology.leaf_nodes, seed=seed,
    )
    orchestrator = EdgeOrchestrator(nodes=topology.all_nodes, catalogue=catalogue)
    orchestrator.load_checkpoint(checkpoint_path)
    orchestrator.agent.epsilon = 0.0
    print(f"  Loaded checkpoint : {checkpoint_path}")
    print(f"  Agent epsilon set : 0.0  (greedy policy)")
    return orchestrator, topology, catalogue, req_gen


def provision_demo(
    checkpoint_path : str  = DEFAULT_CHECKPOINT,
    num_episodes    : int  = 5,
    req_per_ep      : int  = REQUESTS_PER_EPISODE,
    prov_interval   : int  = PROVISIONING_INTERVAL,
    seed            : int  = RANDOM_SEED,
    save_csv        : bool = True,
    traffic_model   : str  = TRAFFIC_MODEL,
    services_csv    : str  = None,
    requests_csv    : str  = None,
) -> None:
    """
    Run num_episodes with the trained policy.
    Prints a per-episode metrics table.
    Saves per-request CSV + text report to results/.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to trained checkpoint
    num_episodes : int
        Number of episodes to run
    req_per_ep : int
        Requests per episode (if not using requests_csv)
    prov_interval : int
        Provisioning interval
    seed : int
        Random seed
    save_csv : bool
        Save results to CSV
    traffic_model : str
        Traffic model for synthetic requests
    services_csv : str, optional
        Path to service catalogue CSV (if None, generates random)
    requests_csv : str, optional
        Path to pre-generated requests CSV (if None, generates synthetic)
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(_SEP)
    print("  Edge Caching  --  Inference / Provision Demo")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Episodes   : {num_episodes}  |  Requests/ep : {req_per_ep}")
    if services_csv:
        print(f"  Services   : {services_csv}")
    if requests_csv:
        print(f"  Requests   : {requests_csv}")
    print(f"  Started    : {ts}")
    print(_DIV)

    # Load orchestrator with optional custom services CSV
    orchestrator, topology, catalogue, req_gen = load_trained_orchestrator(
        checkpoint_path, seed=seed, services_csv=services_csv,
    )
    req_gen.model = traffic_model
    print()

    # Pre-load requests from CSV if provided
    preloaded_requests = None
    if requests_csv:
        preloaded_requests = load_requests_from_csv(requests_csv, catalogue)
        print()

    hdr = (
        f"  {'Ep':>4}  {'Requests':>9}  {'Executed':>9}  "
        f"{'HitRate':>8}  {'Dropped':>8}  {'Avg Latency':>12}  "
        f"{'Leaf%':>7}  {'Cloud%':>7}"
    )
    print(hdr)
    print(_DIV)

    all_rows   : List[dict] = []
    
    # Calculate total_prov based on actual request count
    actual_req_count = len(preloaded_requests) if preloaded_requests else req_per_ep
    total_prov = actual_req_count // prov_interval

    for episode in range(num_episodes):
        # Use preloaded requests or generate synthetic
        if preloaded_requests:
            requests = preloaded_requests
        else:
            requests = req_gen.generate(req_per_ep)
        
        ep_executed   = 0
        ep_forwarded  = 0
        ep_violations = 0
        prov_calls    = 0

        for idx, req in enumerate(requests):
            topology.node_map[req.origin_node_id].receive_request(req, catalogue)
            if req.status == "executed":
                ep_executed += 1
                if req.executed_at_node != req.origin_node_id:
                    ep_forwarded += 1
            elif req.status == "dropped":
                ep_violations += 1
            if (idx + 1) % prov_interval == 0:
                prov_calls += 1
                orchestrator.provision(episode_done=(prov_calls == total_prov))

        s         = _stats(requests)
        leaf_pct  = 100 * s["tiers"].get("leaf",  0) / max(s["total"], 1)
        cloud_pct = 100 * s["tiers"].get("cloud", 0) / max(s["total"], 1)
        print(
            f"  {episode+1:>4}  {s['total']:>9}  {s['executed']:>9}  "
            f"{s['hit_rate']*100:>7.1f}%  {s['dropped']:>8}  "
            f"{s['avg_latency_ms']:>11.2f}ms  {leaf_pct:>6.1f}%  {cloud_pct:>6.1f}%"
        )

        for req in requests:
            all_rows.append(_row(req, catalogue, episode + 1))
        for svc in catalogue.values():
            svc.drift_popularity()

        orchestrator.end_episode(
            episode_reward=0.0, total_executed=ep_executed,
            total_requests=len(requests), total_forwarded=ep_forwarded,
            total_violations=ep_violations, avg_loss=0.0,
        )

    print(_DIV)
    agg_total    = len(all_rows)
    agg_executed = sum(1 for r in all_rows if r["status"] == "executed")
    agg_dropped  = sum(1 for r in all_rows if r["status"] == "dropped")
    print(f"\n  Aggregate over {num_episodes} episodes:")
    print(f"    Total requests  : {agg_total}")
    print(f"    Executed        : {agg_executed}  ({100*agg_executed/max(agg_total,1):.1f}%)")
    print(f"    Dropped         : {agg_dropped}")
    print(_SEP)

    if save_csv:
        results_dir = os.path.join(_ROOT, RESULTS_DIR)
        os.makedirs(results_dir, exist_ok=True)
        csv_path    = os.path.join(results_dir, INFERENCE_CSV)
        report_path = os.path.join(results_dir, INFERENCE_REPORT)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=REQUEST_COLUMNS)
            w.writeheader()
            w.writerows(all_rows)
        print(f"  Request CSV      -> {csv_path}")
        _write_report(
            report_path, ts, checkpoint_path, num_episodes, req_per_ep,
            agg_total, agg_executed, agg_dropped, topology, catalogue,
        )
        print(f"  Summary report   -> {report_path}")


def _write_report(
    path: str, ts: str, checkpoint: str,
    num_episodes: int, req_per_ep: int,
    agg_total: int, agg_executed: int, agg_dropped: int,
    topology: NetworkTopology, catalogue: Dict,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(_SEP + "\n  EDGE CACHING  -  INFERENCE REPORT\n")
        f.write(f"  Timestamp  : {ts}\n  Checkpoint : {checkpoint}\n")
        f.write(_SEP + "\n\n[1] RUN PARAMETERS\n" + _DIV + "\n")
        f.write(f"  Episodes    : {num_episodes}\n  Requests/ep : {req_per_ep}\n")
        f.write(f"  Total reqs  : {agg_total}\n")
        f.write(_DIV + "\n\n[2] OUTCOME SUMMARY\n" + _DIV + "\n")
        hit_rate  = agg_executed / max(agg_total, 1)
        drop_rate = agg_dropped  / max(agg_total, 1)
        f.write(f"  Executed : {agg_executed:<6} ({hit_rate*100:.1f}%)\n")
        f.write(f"  Dropped  : {agg_dropped:<6} ({drop_rate*100:.1f}%)\n")
        f.write(_DIV + "\n\n[3] FINAL CACHE STATE\n" + _DIV + "\n")
        f.write(
            f"  {'Node':<14} {'Level':>6} {'Cap':>5} {'Used%':>7}  Cached Services\n"
        )
        f.write(_DIV + "\n")
        for node in [topology.cloud_root] + topology.all_nodes:
            util = node.cache_utilization()
            f.write(
                f"  {node.node_id:<14} {node.level:>6} {node.cache_cap:>5} "
                f"{util*100:>6.1f}%  {sorted(node.cache)}\n"
            )
        f.write(_DIV + "\n\n[4] SERVICE CATALOGUE\n" + _DIV + "\n")
        f.write(
            f"  {'SvcID':<8} {'Criticality':<18} {'Omega':>6} "
            f"{'Popularity':>11} {'Revenue':>9}\n" + _DIV + "\n"
        )
        for svc_id, svc in sorted(catalogue.items()):
            f.write(
                f"  {svc_id:<8} {svc.criticality:<18} {svc.omega:>6.1f} "
                f"{svc.popularity:>11.4f} {svc.revenue:>9.2f}\n"
            )
        f.write(_DIV + "\n")


def simulate_requests(
    checkpoint_path : str  = DEFAULT_CHECKPOINT,
    num_requests    : int  = REQUESTS_PER_EPISODE,
    seed            : int  = RANDOM_SEED,
    traffic_model   : str  = TRAFFIC_MODEL,
) -> List[dict]:
    """
    Route num_requests using the trained policy.
    Returns a list of per-request result dicts (keys: REQUEST_COLUMNS).
    Designed for notebook or pipeline integration.
    """
    orchestrator, topology, catalogue, req_gen = load_trained_orchestrator(
        checkpoint_path, seed=seed,
    )
    req_gen.model = traffic_model
    requests   = req_gen.generate(num_requests)
    total_prov = num_requests // PROVISIONING_INTERVAL
    prov_calls = 0
    for idx, req in enumerate(requests):
        topology.node_map[req.origin_node_id].receive_request(req, catalogue)
        if (idx + 1) % PROVISIONING_INTERVAL == 0:
            prov_calls += 1
            orchestrator.provision(episode_done=(prov_calls == total_prov))
    return [_row(req, catalogue, 1) for req in requests]


def load_requests_from_csv(csv_path: str, catalogue: Dict) -> List[Request]:
    """
    Load pre-generated requests from CSV file.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file with columns: request_id, service_id, deadline_ms, origin_node_id, arrival_time_ms, cpu_cycles
    catalogue : dict
        Service catalogue to link service IDs
    
    Returns
    -------
    list of Request objects
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    requests = []
    
    for _, row in df.iterrows():
        # Handle origin_node_id as either string (e.g., "L3_23") or int
        origin = str(row["origin_node_id"]) if pd.notna(row["origin_node_id"]) else "0"
        
        req = Request(
            service_id=int(row["service_id"]),
            origin_node_id=origin,
            deadline_ms=float(row.get("deadline_ms", 100.0)),
            arrival_time=float(row.get("arrival_time_ms", 0.0)),
        )
        requests.append(req)
    
    print(f"  Loaded {len(requests)} requests from {csv_path}")
    return requests


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run inference with a trained DDQN edge-caching checkpoint."
    )
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                   help=f"Checkpoint path (default: {DEFAULT_CHECKPOINT})")
    p.add_argument("--episodes",   type=int, default=5,
                   help="Number of evaluation episodes")
    p.add_argument("--requests",   type=int, default=REQUESTS_PER_EPISODE,
                   help="Number of requests per episode (if not using --requests-csv)")
    p.add_argument("--seed",       type=int, default=RANDOM_SEED,
                   help="Random seed for reproducibility")
    p.add_argument("--traffic",    type=str, default=TRAFFIC_MODEL,
                   choices=["poisson_zipf", "uniform"],
                   help="Traffic pattern for synthetic requests")
    p.add_argument("--services",   type=str, default=None,
                   help="Path to service catalogue CSV (if not provided, generates random)")
    p.add_argument("--requests-csv", type=str, default=None,
                   help="Path to pre-generated requests CSV (if not provided, generates synthetic)")
    p.add_argument("--no-csv",     action="store_true",
                   help="Skip saving CSV and report to results/")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    provision_demo(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        req_per_ep=args.requests,
        seed=args.seed,
        save_csv=not args.no_csv,
        traffic_model=args.traffic,
        services_csv=args.services,
        requests_csv=args.requests_csv,
    )
