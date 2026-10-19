import os, csv, random, datetime
from collections import defaultdict
from typing import List, Dict, Tuple
from edge_caching.config import (
    NUM_LEVELS, NUM_CHILDREN, NODE_CACHE_CAP, NUM_SERVICES,
    REQUESTS_PER_EPISODE, PROVISIONING_INTERVAL, RANDOM_SEED, RESULTS_DIR,
)
from edge_caching.core.service import Service, generate_service_catalogue
from edge_caching.core.request import Request
from edge_caching.network.topology import NetworkTopology
from edge_caching.network.cache_initialiser import PlacementReport
from edge_caching.utils.simulation.request_generator import RequestGenerator
from edge_caching.control.orchestrator import EdgeOrchestrator

REQUEST_COLUMNS = [
    "request_id", "service_id", "criticality", "omega", "revenue",
    "origin_node_id", "deadline_ms", "arrival_time",
    "node_arrival_time", "executed_at_node", "remaining_deadline_ms",
    "latency_consumed_ms", "status", "execution_tier",
    "hop_count", "hop_path", "node_log",
]
_W = "=" * 100
_M = "-" * 100


def _tier(req):
    if req.status == "dropped": return "dropped"
    if req.executed_at_node is None: return "unknown"
    if req.executed_at_node == req.origin_node_id: return "leaf"
    if req.executed_at_node == "CLOUD": return "cloud"
    for _, evt in req.node_log:
        if evt == "cache_miss_peer": return "peer"
        if evt in ("cache_miss_parent", "cache_miss_cloud"): return "parent"
    return "parent"


def _latency(req): return req.deadline_ms - req.remaining_deadline


def _build_row(req, catalogue):
    svc = catalogue[req.service_id]
    return {
        "request_id"           : str(req.request_id),
        "service_id"           : str(req.service_id),
        "criticality"          : svc.criticality,
        "omega"                : f"{svc.omega:.1f}",
        "revenue"              : f"{svc.revenue:.2f}",
        "origin_node_id"       : req.origin_node_id,
        "deadline_ms"          : f"{req.deadline_ms:.2f}",
        "arrival_time"         : f"{req.arrival_time:.2f}",
        "node_arrival_time"    : f"{req.node_arrival_time:.2f}",
        "executed_at_node"     : req.executed_at_node or "-",
        "remaining_deadline_ms": f"{req.remaining_deadline:.2f}",
        "latency_consumed_ms"  : f"{_latency(req):.2f}",
        "status"               : req.status,
        "execution_tier"       : _tier(req),
        "hop_count"            : str(len(req.hop_path)),
        "hop_path"             : " -> ".join(f"{n}@{t:.1f}" for n, t in req.hop_path),
        "node_log"             : " | ".join(f"{n}:{e}" for n, e in req.node_log),
    }


def _overall(requests):
    total = len(requests)
    exe   = [r for r in requests if r.status == "executed"]
    drop  = [r for r in requests if r.status == "dropped"]
    local = [r for r in exe if r.executed_at_node == r.origin_node_id]
    fwd   = [r for r in exe if r.executed_at_node != r.origin_node_id]
    tiers = defaultdict(int)
    for r in requests: tiers[_tier(r)] += 1
    return {
        "total": total, "executed": len(exe), "dropped": len(drop),
        "local_hit": len(local), "forwarded_and_hit": len(fwd),
        "hit_rate_pct" : 100 * len(exe)  / max(total, 1),
        "drop_rate_pct": 100 * len(drop) / max(total, 1),
        "tier_counts"  : dict(tiers),
        "avg_remaining_ms": sum(r.remaining_deadline for r in exe) / max(len(exe), 1),
        "avg_latency_ms"  : sum(_latency(r) for r in exe) / max(len(exe), 1),
        "max_hop_count"   : max((len(r.hop_path) for r in requests), default=0),
    }


def _svc_stats(requests, catalogue):
    rows = []
    for svc_id, svc in sorted(catalogue.items()):
        sr  = [r for r in requests if r.service_id == svc_id]
        ex  = [r for r in sr if r.status == "executed"]
        fwd = [r for r in ex if r.executed_at_node != r.origin_node_id]
        dr  = [r for r in sr if r.status == "dropped"]
        rows.append({
            "service_id"    : svc_id,
            "criticality"   : svc.criticality,
            "omega"         : svc.omega,
            "popularity"    : round(svc.popularity, 4),
            "revenue_per"   : svc.revenue,
            "total_reqs"    : len(sr),
            "executed"      : len(ex),
            "forwarded"     : len(fwd),
            "dropped"       : len(dr),
            "hit_rate_pct"  : 100 * len(ex) / max(len(sr), 1),
            "violation_rate": round(svc.sla_violation_rate * 100, 2),
            "total_revenue" : sum(svc.revenue * svc.omega for _ in ex),
        })
    return rows


def _node_stats(requests, topology):
    rows = []
    for node in [topology.cloud_root] + topology.all_nodes:
        exe_here = [r for r in requests if r.executed_at_node == node.node_id]
        fwd_from = [r for r in requests
                    if any(n == node.node_id and e.startswith("cache_miss")
                           for n, e in r.node_log)]
        rows.append({
            "node_id"            : node.node_id,
            "level"              : node.level,
            "cache_cap"          : node.cache_cap,
            "cached_services"    : sorted(node.cache),
            "cache_util_pct"     : round(node.cache_utilization() * 100, 1),
            "requests_executed"  : len(exe_here),
            "requests_forwarded" : len(fwd_from),
            "total_revenue"      : round(node.total_revenue, 4),
        })
    return rows


def _write_init(f, report, catalogue):
    f.write("\n[A] CACHE INITIALISATION  (priority_init strategy)\n")
    f.write(_W + '\n')
    f.write("  Services ranked by: score = omega x revenue\n")
    f.write("  Placement: highest score -> deepest level (leaf first). Each service placed ONCE.\n")
    f.write(_M + '\n')
    f.write(f"  {'Rank':<6} {'SvcID':<8} {'Criticality':<18} {'Omega':>6} {'Revenue':>9} {'Score':>9}  {'Node':<14} {'Level':>6}\n")
    f.write(_M + '\n')
    for rank, (svc_id, score, node_id, level) in enumerate(report.placements, 1):
        svc = catalogue[svc_id]
        f.write(f"  {rank:<6} {svc_id:<8} {svc.criticality:<18} {svc.omega:>6.1f} "
                f"{svc.revenue:>9.2f} {score:>9.3f}  {node_id:<14} {'L'+str(level):>6}\n")
    if report.cloud_only:
        f.write(_M + '\n')
        f.write(f"  Cloud-only: {report.cloud_only}\n")
    f.write(_M + '\n')
    f.write(f"  Placed at edge: {len(report.placements)}  |  Cloud-only: {len(report.cloud_only)}\n")
    f.write(_W + '\n\n')


def _write_requests(f, rows):
    f.write("[B] PER-REQUEST DETAIL LOG\n")
    f.write(_W + '\n')
    fixed = [
        "request_id", "service_id", "criticality", "origin_node_id",
        "deadline_ms", "arrival_time", "node_arrival_time", "executed_at_node",
        "remaining_deadline_ms", "latency_consumed_ms", "status", "execution_tier", "hop_count",
    ]
    FW = {
        "request_id": 12, "service_id": 12, "criticality": 18, "origin_node_id": 14,
        "deadline_ms": 14, "arrival_time": 14, "node_arrival_time": 20,
        "executed_at_node": 18, "remaining_deadline_ms": 22, "latency_consumed_ms": 20,
        "status": 11, "execution_tier": 14, "hop_count": 11,
    }
    f.write("".join(c.ljust(FW[c]) for c in fixed) + "\n")
    f.write(_M + '\n')
    for row in rows:
        f.write("".join(row[c].ljust(FW[c]) for c in fixed) + "\n")
        f.write(f"    hop_path : {row['hop_path']}\n")
        f.write(f"    node_log : {row['node_log']}\n")
    f.write(_W + '\n\n')


def _write_overall(f, s):
    f.write("[C] OVERALL SUMMARY STATISTICS\n")
    f.write(_W + '\n')
    f.write(f"  Total requests            : {s['total']}\n\n")
    f.write("  -- Outcome Counts --\n")
    f.write(f"  Successfully executed     : {s['executed']:<6}  ({s['hit_rate_pct']:.1f}%)\n")
    f.write(f"  Deadline expired/dropped  : {s['dropped']:<6}  ({s['drop_rate_pct']:.1f}%)\n\n")
    f.write("  -- Hit Location --\n")
    f.write(f"  Local cache hit           : {s['local_hit']}\n")
    f.write(f"  Forwarded then executed   : {s['forwarded_and_hit']}\n\n")
    f.write("  -- Latency --\n")
    f.write(f"  Avg remaining deadline    : {s['avg_remaining_ms']:.2f} ms\n")
    f.write(f"  Avg latency consumed      : {s['avg_latency_ms']:.2f} ms\n")
    f.write(f"  Max hop count             : {s['max_hop_count']}\n")
    f.write(_W + '\n\n')


def _write_tier(f, s):
    f.write("[D] EXECUTION-TIER BREAKDOWN\n")
    f.write(_W + '\n')
    total = s['total']
    labels = {
        "leaf"   : "Executed at leaf   (direct cache hit)",
        "peer"   : "Executed at peer    (same-level forward)",
        "parent" : "Executed at parent/ancestor edge node",
        "cloud"  : "Executed at cloud root (full escalation)",
        "dropped": "Dropped  (deadline exceeded during routing)",
    }
    f.write(f"  {'Tier':<44} {'Count':>8}  {'%':>7}\n")
    f.write(_M + '\n')
    for tier, label in labels.items():
        cnt = s['tier_counts'].get(tier, 0)
        f.write(f"  {label:<44} {cnt:>8}  {100*cnt/max(total,1):>6.1f}%\n")
    f.write(_W + '\n\n')


def _write_service(f, rows):
    f.write("[E] PER-SERVICE STATISTICS\n")
    f.write(_W + '\n')
    f.write(
        f"  {'SvcID':<8} {'Criticality':<18} {'Omega':>6} {'Popularity':>11} "
        f"{'Rev/exe':>8} {'Requests':>10} {'Executed':>10} "
        f"{'Forwarded':>11} {'Dropped':>9} {'HitRate%':>10} "
        f"{'SLA-Viol%':>10} {'Revenue':>10}\n"
    )
    f.write(_M + '\n')
    for r in rows:
        f.write(
            f"  {r['service_id']:<8} {r['criticality']:<18} {r['omega']:>6.1f} "
            f"{r['popularity']:>11.4f} {r['revenue_per']:>8.2f} "
            f"{r['total_reqs']:>10} {r['executed']:>10} "
            f"{r['forwarded']:>11} {r['dropped']:>9} "
            f"{r['hit_rate_pct']:>9.1f}% {r['violation_rate']:>9.2f}% "
            f"{r['total_revenue']:>10.3f}\n"
        )
    f.write(_M + '\n')
    f.write(
        f"  {'TOTAL':<56} "
        f"{sum(r['total_reqs'] for r in rows):>10} "
        f"{sum(r['executed'] for r in rows):>10} "
        f"{sum(r['forwarded'] for r in rows):>11} "
        f"{sum(r['dropped'] for r in rows):>9} "
        f"{'':>10} {'':>10} "
        f"{sum(r['total_revenue'] for r in rows):>10.3f}\n"
    )
    f.write(_W + '\n\n')


def _write_node(f, rows):
    f.write("[F] PER-NODE STATISTICS\n")
    f.write(_W + '\n')
    f.write(
        f"  {'Node':<14} {'Lvl':>4} {'Cap':>5} {'Used%':>7} "
        f"{'Executed':>10} {'Forwarded':>11} {'Revenue':>10}  Cached Services\n"
    )
    f.write(_M + '\n')
    for r in rows:
        f.write(
            f"  {r['node_id']:<14} {r['level']:>4} {r['cache_cap']:>5} "
            f"{r['cache_util_pct']:>6.1f}% {r['requests_executed']:>10} "
            f"{r['requests_forwarded']:>11} {r['total_revenue']:>10.4f}  "
            f"{r['cached_services']}\n"
        )
    f.write(_W + '\n\n')


def run(
    num_levels=NUM_LEVELS, num_children=NUM_CHILDREN, cache_cap=NODE_CACHE_CAP,
    num_services=NUM_SERVICES, req_per_ep=REQUESTS_PER_EPISODE,
    prov_interval=PROVISIONING_INTERVAL, seed=RANDOM_SEED,
):
    random.seed(seed)
    cat_list  = generate_service_catalogue(num_services, seed=seed)
    catalogue = {s.service_id: s for s in cat_list}
    topology  = NetworkTopology(num_levels=num_levels, num_children=num_children, cache_cap=cache_cap)
    topology.prime_cloud_cache(catalogue)
    report       = topology.priority_init(catalogue)
    req_gen      = RequestGenerator(catalogue=catalogue, leaf_nodes=topology.leaf_nodes, seed=seed)
    orchestrator = EdgeOrchestrator(nodes=topology.all_nodes, catalogue=catalogue)
    requests = req_gen.generate(req_per_ep)
    for idx, req in enumerate(requests):
        topology.node_map[req.origin_node_id].receive_request(req, catalogue)
        if (idx + 1) % prov_interval == 0:
            orchestrator.provision()
    return requests, report, topology, catalogue


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = os.path.join(RESULTS_DIR, "run_log.txt")
    log_path_csv = os.path.join(RESULTS_DIR, "run_log.csv")
    print(_W); print('  Edge Caching Simulation Runner'); print(f'  {ts}'); print(_W)
    print('\n[1/4] Running simulation ...')
    requests, report, topology, catalogue = run()
    print('[2/4] Building statistics ...')
    rows      = [_build_row(r, catalogue) for r in requests]
    s_overall = _overall(requests)
    s_svc     = _svc_stats(requests, catalogue)
    s_node    = _node_stats(requests, topology)
    print('[3/4] Writing run log ...')
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(_W + '\n  EDGE CACHING SIMULATION  -  RUN LOG\n')
        f.write(f"  Run timestamp  : {ts}\n")
        f.write(f"  Topology       : {NUM_LEVELS} levels, {NUM_CHILDREN} children, cache_cap={NODE_CACHE_CAP}\n")
        f.write(f"  Services       : {NUM_SERVICES}  Requests: {REQUESTS_PER_EPISODE}  Prov interval: {PROVISIONING_INTERVAL}\n")
        f.write(f"  Random seed    : {RANDOM_SEED}\n" + _W + "\n")
        f.write("\n[TOPOLOGY]\n" + _M + "\n" + topology.summary() + "\n" + _M + "\n\n")
        _write_init(f, report, catalogue)
        _write_requests(f, rows)
        _write_overall(f, s_overall)
        _write_tier(f, s_overall)
        _write_service(f, s_svc)
        _write_node(f, s_node)
        f.write(f"\n  [END OF RUN LOG]  {ts}\n" + _W + "\n")
    print('[4/4] Writing CSV ...')
    with open(log_path_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REQUEST_COLUMNS)
        w.writeheader(); w.writerows(rows)
    print()
    print(_W); print('  RESULTS SUMMARY'); print(_W)
    print(f"  Total requests          : {s_overall['total']}")
    print(f"  Successfully executed   : {s_overall['executed']:<6}  ({s_overall['hit_rate_pct']:.1f}%)")
    print(f"  Deadline expired/dropped: {s_overall['dropped']:<6}  ({s_overall['drop_rate_pct']:.1f}%)")
    print('  -- Execution Tier --')
    for tier, label in {'leaf':'Local leaf hit','peer':'Peer forward','parent':'Parent forward','cloud':'Cloud fallback','dropped':'Dropped'}.items():
        cnt = s_overall['tier_counts'].get(tier, 0)
        print(f"    {label:<22}: {cnt:>5}  ({100*cnt/max(s_overall['total'],1):.1f}%)")
    print(f"  Avg remaining deadline  : {s_overall['avg_remaining_ms']:.2f} ms")
    print(f"  Avg hop latency         : {s_overall['avg_latency_ms']:.2f} ms")
    print(_W)
    print(f'  Run log  -> {log_path}')
    print(f'  CSV      -> {log_path_csv}')
    print(_W)


if __name__ == "__main__":
    main()