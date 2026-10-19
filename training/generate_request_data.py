import os
import csv
import random
from typing import List, Dict
from edge_caching.config import (
    NUM_LEVELS, NUM_CHILDREN, NODE_CACHE_CAP,
    NUM_SERVICES, REQUESTS_PER_EPISODE, PROVISIONING_INTERVAL,
    RANDOM_SEED, RESULTS_DIR,
)
from edge_caching.core.service import generate_service_catalogue
from edge_caching.core.request import Request
from edge_caching.network.topology import NetworkTopology
from edge_caching.utils.simulation.request_generator import RequestGenerator
from edge_caching.control.orchestrator import EdgeOrchestrator

COLUMNS = [
    "request_id", "service_id", "origin_node_id", "deadline_ms",
    "arrival_time", "node_arrival_time", "executed_at_node",
    "remaining_deadline_ms", "status", "hop_count", "hop_path", "node_log",
]
W = {
    "request_id": 12, "service_id": 12, "origin_node_id": 16,
    "deadline_ms": 14, "arrival_time": 14, "node_arrival_time": 20,
    "executed_at_node": 18, "remaining_deadline_ms": 22,
    "status": 10, "hop_count": 12, "hop_path": 55, "node_log": 65,
}


def _row(req: Request) -> Dict[str, str]:
    hop_str = " -> ".join(f"{n}@{t:.1f}" for n, t in req.hop_path)
    log_str = " | ".join(f"{n}:{e}" for n, e in req.node_log)
    return {
        "request_id":            str(req.request_id),
        "service_id":            str(req.service_id),
        "origin_node_id":        req.origin_node_id,
        "deadline_ms":           f"{req.deadline_ms:.2f}",
        "arrival_time":          f"{req.arrival_time:.2f}",
        "node_arrival_time":     f"{req.node_arrival_time:.2f}",
        "executed_at_node":      req.executed_at_node or "-",
        "remaining_deadline_ms": f"{req.remaining_deadline:.2f}",
        "status":                req.status,
        "hop_count":             str(len(req.hop_path)),
        "hop_path":              hop_str,
        "node_log":              log_str,
    }


def _print_timing_legend() -> None:
    print("""
Timing Model  (Strategy B - Cumulative Latency)
================================================
  arrival_time       : simulated clock when request was GENERATED at origin
                       leaf node. NEVER changes across hops.
  node_arrival_time  : arrival_time + total_hop_latency_consumed_so_far
                       re-derived after EVERY forwarding hop automatically.
  remaining_deadline : deadline_ms - total_hop_latency_consumed_so_far
                       reflects the budget still available at the executing node.
  hop_path           : timing trace  - node_id@arrival_time at each node
  node_log           : routing trace - node_id:decision at each node
                       decisions: cache_hit | cache_miss_peer |
                                  cache_miss_parent | cache_miss_cloud |
                                  executed | dropped
""")


def _print_table(records: List[Dict[str, str]]) -> None:
    fixed_cols = [c for c in COLUMNS if c not in ("hop_path", "node_log")]
    hdr = "".join(c.ljust(W[c]) for c in fixed_cols)
    div = "-" * len(hdr)
    print(div)
    print(hdr)
    print(div)
    for rec in records:
        print("".join(rec[c].ljust(W[c]) for c in fixed_cols))
        print("    hop_path :", rec["hop_path"])
        print("    node_log :", rec["node_log"])
    print(div)
    total    = len(records)
    executed = sum(1 for r in records if r["status"] == "executed")
    dropped  = sum(1 for r in records if r["status"] == "dropped")
    pending  = sum(1 for r in records if r["status"] == "pending")
    print(f"  Total: {total}  Executed: {executed}  Dropped: {dropped}  Pending: {pending}")
    print(div)


def _save_csv(records: List[Dict[str, str]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    print(f"CSV saved -> {path}")


def run_and_collect(
    num_levels    = NUM_LEVELS,
    num_children  = NUM_CHILDREN,
    cache_cap     = NODE_CACHE_CAP,
    num_services  = NUM_SERVICES,
    req_per_ep    = REQUESTS_PER_EPISODE,
    prov_interval = PROVISIONING_INTERVAL,
    seed          = RANDOM_SEED,
) -> List[Request]:
    """Build topology, run priority_init, route one episode of requests."""
    random.seed(seed)
    cat_list  = generate_service_catalogue(num_services, seed=seed)
    catalogue = {s.service_id: s for s in cat_list}
    topology  = NetworkTopology(
        num_levels=num_levels, num_children=num_children, cache_cap=cache_cap
    )
    topology.prime_cloud_cache(catalogue)
    report = topology.priority_init(catalogue)
    req_gen = RequestGenerator(
        catalogue=catalogue, leaf_nodes=topology.leaf_nodes, seed=seed
    )
    orchestrator = EdgeOrchestrator(
        nodes=topology.all_nodes, catalogue=catalogue
    )
    print("=== Topology ===")
    print(topology.summary())
    print()
    report.print_report()
    requests: List[Request] = req_gen.generate(req_per_ep)
    print(f"Routing {len(requests)} requests through the topology ...\n")
    for idx, req in enumerate(requests):
        topology.node_map[req.origin_node_id].receive_request(req, catalogue)
        if (idx + 1) % prov_interval == 0:
            orchestrator.provision()
    return requests


def main():
    print("=" * 90)
    print("  Edge Caching - Request Data Generator  (Cumulative Latency + Node Logic)")
    print("=" * 90)
    _print_timing_legend()
    requests = run_and_collect()
    records  = [_row(r) for r in requests]
    print("=== Request Log ===\n")
    _print_table(records)
    csv_path = os.path.join(RESULTS_DIR, "request_data.csv")
    _save_csv(records, csv_path)


if __name__ == "__main__":
    main()
