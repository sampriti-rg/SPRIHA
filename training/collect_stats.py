"""
collect_stats.py
----------------
Phase 1 of the realistic-data training pipeline.

Purpose
-------
Run a warm-up simulation using SAMPLE SERVICE DATA loaded from
data/sample_services.csv instead of the purely random catalogue.
At every provisioning interval, record the raw per-(service, node)
statistics that the DDQN uses to build its state vector.

These collected statistics are saved to:
    data/collected_stats.csv      -- one row per (episode, interval, node, service)
    data/collected_states.csv     -- the 8-element normalised state vectors
    data/collected_transitions.csv-- full (s, a, r, s', done) transitions

The collected data feeds directly into finetune.py which uses it to:
  1. Pre-populate the replay buffer with realistic transitions
  2. Continue training the DDQN from a real-data-informed starting point

What makes this more realistic than pure training
-------------------------------------------------
The random catalogue in generate_service_catalogue() assigns:
  - popularity : Uniform(0.1, 1.0)  -- no structure
  - revenue    : Uniform(1.0, 10.0) -- no structure
  - criticality: round-robin tiers  -- no domain knowledge

The sample_services.csv assigns values that reflect real edge service
characteristics:
  - VideoStreaming: high popularity, high revenue, mission_critical
  - BackupService : low popularity, low revenue, best_effort
  - VoIP          : moderate popularity, highest revenue, mission_critical
  etc.

This means the collected forwarding rates, cache utilization, and
violation rates are grounded in plausible service demand patterns.

Usage
-----
    python collect_stats.py                    # uses data/sample_services.csv
    python collect_stats.py --episodes 50      # longer warm-up
    python collect_stats.py --services data/my_services.csv
"""

import argparse
import csv
import os
import random
import sys
from typing import Dict, List

import numpy as np

# Ensure the project root is on sys.path when run as  python training/collect_stats.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge_caching.config import (
    NUM_LEVELS, NUM_CHILDREN, NODE_CACHE_CAP,
    REQUESTS_PER_EPISODE, PROVISIONING_INTERVAL, RANDOM_SEED,
    TRAFFIC_MODEL, HOP_LATENCY_MS,
)
from edge_caching.core.service import Service
from edge_caching.core.request import Request
from edge_caching.network.topology import NetworkTopology
from edge_caching.utils.simulation.request_generator import RequestGenerator
from edge_caching.control.orchestrator import EdgeOrchestrator
from edge_caching.agents.ddqn_agent import DDQNAgent

_ROOT                 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR              = os.path.join(_ROOT, "training", "data")
SAMPLE_SERVICES_CSV   = os.path.join(DATA_DIR, "sample_services.csv")
COLLECTED_STATS_CSV   = os.path.join(DATA_DIR, "collected_stats.csv")
COLLECTED_STATES_CSV  = os.path.join(DATA_DIR, "collected_states.csv")
COLLECTED_TRANS_CSV   = os.path.join(DATA_DIR, "collected_transitions.csv")

_SEP = "=" * 72
_DIV = "-" * 72

# CSV column definitions
STATS_COLUMNS = [
    "episode", "interval", "node_id", "node_level",
    "service_id", "service_name", "criticality", "omega",
    "size", "popularity", "revenue",
    "forwarded_count", "executed_count",
    "cache_utilization", "violation_count", "demand_count",
    "sla_violation_rate",
]

STATE_COLUMNS = [
    "episode", "interval", "node_id", "service_id",
    "s0_size_norm", "s1_popularity", "s2_omega_norm", "s3_revenue_norm",
    "s4_forwarded_rate", "s5_executed_rate", "s6_cache_util", "s7_sla_viol_rate",
    "s8_is_cached", "s9_slack_norm",
]

TRANSITION_COLUMNS = [
    "episode", "interval", "node_id", "service_id",
    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9",
    "action", "reward",
    "ns0", "ns1", "ns2", "ns3", "ns4", "ns5", "ns6", "ns7", "ns8", "ns9",
    "done",
]


# ---------------------------------------------------------------------------
# Load sample services from CSV
# ---------------------------------------------------------------------------

def load_sample_services(csv_path: str) -> Dict[int, Service]:
    """
    Load service definitions from a CSV file.

    Expected columns (see data/sample_services.csv):
        service_id, name, size, popularity, criticality, revenue, description

    Returns
    -------
    {service_id: Service}
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Sample services file not found: {csv_path}\n"
            f"Expected columns: service_id, name, size, popularity, "
            f"criticality, revenue, description"
        )

    catalogue: Dict[int, Service] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            svc = Service(
                service_id  = int(row["service_id"]),
                size        = int(row["size"]),
                popularity  = float(row["popularity"]),
                criticality = row["criticality"].strip(),
                revenue     = float(row["revenue"]),
            )
            catalogue[svc.service_id] = svc

    print(f"  Loaded {len(catalogue)} services from {csv_path}")
    for svc_id, svc in sorted(catalogue.items()):
        name = row.get("name", f"svc_{svc_id}")  # last row name for display
        print(f"    [{svc_id}] {svc.criticality:<18} pop={svc.popularity:.2f}  "
              f"rev={svc.revenue:.2f}  omega={svc.omega:.1f}")
    return catalogue


def load_sample_services_with_names(csv_path: str):
    """
    Load services and return (catalogue_dict, name_lookup_dict).
    Reads the optional init_cpu_cycles column.  Falls back to 50000
    when the column is absent so older CSV files remain compatible.
    """
    catalogue: Dict[int, Service] = {}
    names: Dict[int, str] = {}
    with open(csv_path, newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            svc_id = int(row["service_id"])
            svc = Service(
                service_id      = svc_id,
                size            = int(row["size"]),
                popularity      = float(row["popularity"]),
                criticality     = row["criticality"].strip(),
                revenue         = float(row["revenue"]),
                init_cpu_cycles = float(row["init_cpu_cycles"])
                                  if "init_cpu_cycles" in row and row["init_cpu_cycles"]
                                  else 50000,
            )
            catalogue[svc_id] = svc
            names[svc_id] = row.get("name", "service_{}".format(svc_id)).strip()
    return catalogue, names


# ---------------------------------------------------------------------------
# Stat row builders
# ---------------------------------------------------------------------------

def _build_stats_row(
    episode    : int,
    interval   : int,
    node_id    : str,
    node_level : int,
    svc        : Service,
    svc_name   : str,
    snap       : dict,
) -> dict:
    svc_id = svc.service_id
    return {
        "episode"          : episode,
        "interval"         : interval,
        "node_id"          : node_id,
        "node_level"       : node_level,
        "service_id"       : svc_id,
        "service_name"     : svc_name,
        "criticality"      : svc.criticality,
        "omega"            : f"{svc.omega:.1f}",
        "size"             : svc.size,
        "popularity"       : f"{svc.popularity:.4f}",
        "revenue"          : f"{svc.revenue:.2f}",
        "forwarded_count"  : snap["forwarded_count"].get(svc_id, 0),
        "executed_count"   : snap["executed_count"].get(svc_id, 0),
        "cache_utilization": f"{snap['cache_utilization']:.4f}",
        "violation_count"  : svc.violation_count,
        "demand_count"     : svc.demand_count,
        "sla_violation_rate": f"{svc.sla_violation_rate:.4f}",
    }


def _build_state_row(
    episode   : int,
    interval  : int,
    node_id   : str,
    svc       : Service,
    snap      : dict,
    vec       : np.ndarray,
) -> dict:
    return {
        "episode"    : episode,
        "interval"   : interval,
        "node_id"    : node_id,
        "service_id" : svc.service_id,
        "s0_size_norm"       : f"{vec[0]:.6f}",
        "s1_popularity"      : f"{vec[1]:.6f}",
        "s2_omega_norm"      : f"{vec[2]:.6f}",
        "s3_revenue_norm"    : f"{vec[3]:.6f}",
        "s4_forwarded_rate"  : f"{vec[4]:.6f}",
        "s5_executed_rate"   : f"{vec[5]:.6f}",
        "s6_cache_util"      : f"{vec[6]:.6f}",
        "s7_sla_viol_rate"   : f"{vec[7]:.6f}",
        "s8_is_cached"       : f"{vec[8]:.6f}",
        "s9_slack_norm"      : f"{vec[9]:.6f}",
    }


def _build_transition_row(
    episode   : int,
    interval  : int,
    node_id   : str,
    svc_id    : int,
    state     : np.ndarray,
    action    : int,
    reward    : float,
    next_state: np.ndarray,
    done      : bool,
) -> dict:
    row = {
        "episode"   : episode,
        "interval"  : interval,
        "node_id"   : node_id,
        "service_id": svc_id,
        "action"    : action,
        "reward"    : f"{reward:.6f}",
        "done"      : int(done),
    }
    for i, v in enumerate(state):
        row[f"s{i}"] = f"{v:.6f}"
    for i, v in enumerate(next_state):
        row[f"ns{i}"] = f"{v:.6f}"
    return row


# ---------------------------------------------------------------------------
# Streaming CSV writer helpers
# ---------------------------------------------------------------------------

def _open_csv_writers(data_dir: str):
    """
    Open ONLY the transitions CSV for incremental (streaming) writes.
    
    collected_stats.csv and collected_states.csv are NOT written because:
      - finetune.py only reads collected_transitions.csv
      - At 50 episodes they consume ~2.4 GB each with zero benefit
      - Removing them reduces disk use from ~218 MB/ep to ~60 MB/ep
    
    Returns (trans_f, trans_w).
    Caller must close the file handle when done.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    trans_f = open(os.path.join(data_dir, "collected_transitions.csv"),
                   "w", newline="", encoding="utf-8")
    trans_w = csv.DictWriter(trans_f, fieldnames=TRANSITION_COLUMNS)
    trans_w.writeheader()

    return trans_f, trans_w


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect(
    services_csv  : str = SAMPLE_SERVICES_CSV,
    num_episodes  : int = 30,
    req_per_ep    : int = REQUESTS_PER_EPISODE,
    prov_interval : int = PROVISIONING_INTERVAL,
    seed          : int = RANDOM_SEED,
    traffic_model : str = TRAFFIC_MODEL,
    num_levels    : int = NUM_LEVELS,
    num_children  : int = NUM_CHILDREN,
    cache_cap     : int = NODE_CACHE_CAP,
) -> None:
    """
    Run the warm-up simulation and collect statistics.

    The agent here uses EPSILON=1.0 throughout (pure random actions) so
    all three actions (INSERT / DELETE / KEEP) are sampled uniformly,
    producing a diverse set of realistic transitions that cover all regions
    of the state space.

    Outputs
    -------
    data/collected_stats.csv       per-(episode, interval, node, service) statistics
    data/collected_states.csv      8-element normalised state vectors
    data/collected_transitions.csv full (s, a, r, s', done) transitions
    """
    random.seed(seed)
    np.random.seed(seed)

    ts_start = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(_SEP)
    print("  Edge Caching  --  Statistics Collection")
    print(f"  Services CSV  : {services_csv}")
    print(f"  Episodes      : {num_episodes}  |  Req/ep : {req_per_ep}")
    print(f"  Started       : {ts_start}")
    print(_DIV)

    # ?? Load realistic service catalogue ?????????????????????????????????
    catalogue, names = load_sample_services_with_names(services_csv)
    print()

    # ?? Build topology ????????????????????????????????????????????????????
    topology = NetworkTopology(
        num_levels=num_levels, num_children=num_children, cache_cap=cache_cap,
    )
    topology.prime_cloud_cache(catalogue)
    topology.priority_init(catalogue)

    # ?? Request generator using realistic popularities ????????????????????
    req_gen = RequestGenerator(
        catalogue=catalogue, leaf_nodes=topology.leaf_nodes,
        seed=seed, model=traffic_model,
    )

    # ?? Orchestrator: agent runs with EPSILON=1.0 for full coverage ???????
    orchestrator = EdgeOrchestrator(nodes=topology.all_nodes, catalogue=catalogue)
    orchestrator.agent.epsilon = 1.0   # pure random actions ? diverse transitions

    # Open transitions CSV only (stats/states files not needed by finetune.py)
    trans_f, trans_w = _open_csv_writers(DATA_DIR)

    total_trans_rows = 0

    total_prov = req_per_ep // prov_interval

    print(f"\n  Collecting over {num_episodes} episodes ...\n")
    print(f"  {'Ep':>4}  {'Requests':>9}  {'Executed':>9}  {'HitRate':>8}  "
          f"{'Transitions':>12}  {'Intervals':>10}")
    print(_DIV)

    for episode in range(num_episodes):
        requests      = req_gen.generate(req_per_ep)
        ep_executed   = 0
        ep_violations = 0
        prov_calls    = 0
        ep_transitions = 0

        for idx, req in enumerate(requests):
            topology.node_map[req.origin_node_id].receive_request(req, catalogue)
            if req.status == "executed":
                ep_executed += 1
            elif req.status == "dropped":
                ep_violations += 1

            if (idx + 1) % prov_interval == 0:
                prov_calls += 1
                interval_num = episode * total_prov + prov_calls
                is_last      = (prov_calls == total_prov)

                # Capture (s, a, r, s', done) for each (node, service) pair
                # and write directly to CSV. Then call provision() to maintain
                # the orchestrator's internal state (prev_snapshots, EMA, etc.)
                # NOTE: provision() also calls store_transition() internally
                # that goes into the replay buffer only, NOT into the CSV.
                pre_snapshots = {
                    node.node_id: node.get_stats_snapshot()
                    for node in topology.all_nodes
                }

                for node in topology.all_nodes:
                    snap = pre_snapshots[node.node_id]
                    for svc_id, svc in catalogue.items():
                        state_vec  = DDQNAgent.build_state_vector(svc, snap)
                        action     = orchestrator.agent.select_action(state_vec)
                        cost       = node.apply_action(action, svc)
                        prev_snap  = orchestrator._prev_snapshots.get(node.node_id, {})
                        delta_exe  = max(0, snap["executed_count"].get(svc_id, 0)
                                        - prev_snap.get("executed_count", {}).get(svc_id, 0))
                        delta_fwd  = max(0, snap["forwarded_count"].get(svc_id, 0)
                                        - prev_snap.get("forwarded_count", {}).get(svc_id, 0))
                        reward     = DDQNAgent.compute_reward(
                            service=svc,
                            delta_served=delta_exe,
                            delta_executed=delta_exe,
                            delta_forwarded=delta_fwd,
                            placement_cost=cost,
                        )
                        next_state_vec = DDQNAgent.build_state_vector(
                            svc, node.get_stats_snapshot()
                        )
                        trans_w.writerow(_build_transition_row(
                            episode=episode + 1, interval=interval_num,
                            node_id=node.node_id, svc_id=svc_id,
                            state=state_vec, action=action, reward=reward,
                            next_state=next_state_vec, done=is_last,
                        ))
                        ep_transitions += 1
                        total_trans_rows += 1

                # Maintain orchestrator internal state (prev_snapshots, EMA,
                # interval counters). Agent learns are suppressed by epsilon=1.0.
                orchestrator.provision(episode_done=is_last)

        # ?? Episode boundary ??????????????????????????????????????????????
        for svc in catalogue.values():
            svc.drift_popularity()
        orchestrator.end_episode(
            episode_reward=0.0, total_executed=ep_executed,
            total_requests=req_per_ep, total_forwarded=0,
            total_violations=ep_violations, avg_loss=0.0,
        )

        hit_rate = ep_executed / max(req_per_ep, 1)
        print(
            f"  {episode+1:>4}  {req_per_ep:>9}  {ep_executed:>9}  "
            f"{hit_rate*100:>7.1f}%  {ep_transitions:>12}  {prov_calls:>10}"
        )
        # Flush to disk after each episode so data is safe if interrupted
        trans_f.flush()

    # Close CSV file handle
    trans_f.close()

    print(_SEP)
    print("  Collection complete")
    print(f"  Transition rows  : {total_trans_rows:>7}  -> {COLLECTED_TRANS_CSV}")
    print(_SEP)
    print()
    print("  Next step:  python finetune.py")
    print()


def _write_csv(path: str, columns: List[str], rows: List[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows):>7} rows -> {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect realistic statistics for DDQN fine-tuning."
    )
    p.add_argument("--services",  type=str, default=SAMPLE_SERVICES_CSV,
                   help=f"Path to sample services CSV (default: {SAMPLE_SERVICES_CSV})")
    p.add_argument("--episodes",  type=int, default=30,
                   help="Warm-up episodes to run (default: 30)")
    p.add_argument("--requests",  type=int, default=REQUESTS_PER_EPISODE,
                   help=f"Requests per episode (default: {REQUESTS_PER_EPISODE})")
    p.add_argument("--seed",      type=int, default=RANDOM_SEED,
                   help=f"Random seed (default: {RANDOM_SEED})")
    p.add_argument("--traffic",   type=str, default=TRAFFIC_MODEL,
                   choices=["poisson_zipf", "uniform"],
                   help="Traffic model (default: poisson_zipf)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    collect(
        services_csv  = args.services,
        num_episodes  = args.episodes,
        req_per_ep    = args.requests,
        seed          = args.seed,
        traffic_model = args.traffic,
    )
