"""
train.py
--------
End-to-end DDQN training script for edge cache provisioning.

Responsibilities
----------------
1.  Seed all random sources for reproducibility.
2.  Build the service catalogue and N-Ary edge topology.
3.  Warm-start caches with priority_init so the agent starts from a
    meaningful state rather than all-empty caches.
4.  Run the full multi-episode training loop via Simulation.run().
5.  Save the trained DDQN checkpoint (online_net, target_net, optimizer,
    epsilon, step counters) to  checkpoints/ddqn_final.pt
6.  Export a lightweight inference-only copy of the online network weights
    to  checkpoints/ddqn_online_weights.pt  for use in the request
    simulator without importing the full agent.
7.  Persist per-episode training metrics to  results/training_metrics.csv
    so convergence can be inspected without re-running the simulation.
8.  Save all visualisation plots (reward curve, hit rate, loss, etc.)
    to  results/  via save_all_results().

Usage
-----
    python train.py                          # default config from config.py
    python train.py --episodes 300           # override episode count
    python train.py --no-plots               # skip visualisation (faster CI)
    python train.py --checkpoint my_run.pt   # custom checkpoint filename

The script is safe to re-run: checkpoints directory is created if absent
and any existing checkpoint is overwritten.
"""

import argparse
import csv
import os
import random
import datetime
import sys

# Ensure the project root is on sys.path when the script is run directly
# from the training/ sub-directory or via  python training/train.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from edge_caching.config import (
    NUM_EPISODES, NUM_LEVELS, NUM_CHILDREN, NODE_CACHE_CAP,
    NUM_SERVICES, REQUESTS_PER_EPISODE, PROVISIONING_INTERVAL,
    RANDOM_SEED, RESULTS_DIR, TRAFFIC_MODEL,CURRICULUM_SCHEDULE,
    USE_CURRICULUM_LEARNING, USE_VARIABLE_LOAD, 
    LOAD_DISTRIBUTION, LOAD_WEIGHTS
)
from edge_caching.core.service import generate_service_catalogue
import csv as _csv
from edge_caching.network.topology import NetworkTopology
from edge_caching.utils.simulation.request_generator import RequestGenerator
from edge_caching.control.orchestrator import EdgeOrchestrator
from edge_caching.utils.visualiser import save_all_results

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Paths are always relative to the project root, not the training/ sub-folder
_ROOT               = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR      = os.path.join(_ROOT, "checkpoints")
CHECKPOINT_FILE     = "ddqn_final.pt"
ONLINE_WEIGHTS_FILE = "ddqn_online_weights.pt"
METRICS_CSV         = "training_metrics.csv"

_SEP = "=" * 72
_DIV = "-" * 72


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_all(seed: int) -> None:
    """Pin every random source so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_metrics(orchestrator: EdgeOrchestrator, path: str) -> None:
    """Write per-episode training metrics to a CSV file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fields = [
        "episode", "reward", "profit", "hit_rate",
        "forwarded", "violations", "avg_loss", "epsilon",
    ]
    rows = []
    epsilon_series = _reconstruct_epsilon_series(orchestrator)
    for ep in range(len(orchestrator.episode_rewards)):
        rows.append({
            "episode"   : ep + 1,
            "reward"    : round(orchestrator.episode_rewards[ep],    4),
            "profit"    : round(orchestrator.episode_profits[ep],    4),
            "hit_rate"  : round(orchestrator.episode_hit_rates[ep],  4),
            "forwarded" : orchestrator.episode_fwd_counts[ep],
            "violations": orchestrator.episode_violations[ep],
            "avg_loss"  : round(orchestrator.episode_losses[ep],     6),
            "epsilon"   : round(epsilon_series[ep],                  6),
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _reconstruct_epsilon_series(orchestrator: EdgeOrchestrator):
    """
    Reconstruct per-episode epsilon values from the agent's final state.
    Works backward from the current epsilon using the EPSILON_DECAY schedule.
    """
    from edge_caching.config import EPSILON_START, EPSILON_MIN, EPSILON_DECAY
    n = len(orchestrator.episode_rewards)
    eps = orchestrator.agent.epsilon
    series = [eps] * n
    # Unwind backwards to reconstruct the full schedule
    for i in range(n - 2, -1, -1):
        prev = series[i + 1] / EPSILON_DECAY
        series[i] = min(EPSILON_START, max(EPSILON_MIN, prev))
    return series


def _print_episode_summary(
    ep: int,
    total: int,
    reward: float,
    hit_rate: float,
    violations: int,
    avg_loss: float,
    epsilon: float,
) -> None:
    print(
        f"  Ep {ep+1:>4}/{total}  |  "
        f"Reward: {reward:>8.3f}  |  "
        f"HitRate: {hit_rate*100:>5.1f}%  |  "
        f"Violations: {violations:>4}  |  "
        f"Loss: {avg_loss:>8.5f}  |  "
        f"?: {epsilon:.4f}"
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _load_services_csv(csv_path: str) -> dict:
    """Load service catalogue from a CSV file (sample_services.csv format)."""
    from edge_caching.core.service import Service
    catalogue = {}
    with open(csv_path, newline="", encoding="utf-8-sig", errors="replace") as f:
        for row in _csv.DictReader(f):
            from edge_caching.config import CRITICALITY_WEIGHTS
            svc = Service(
                service_id      = int(row["service_id"]),
                size            = float(row["size"]),
                popularity      = float(row["popularity"]),
                criticality     = row["criticality"].strip(),
                revenue         = float(row["revenue"]),
                init_cpu_cycles = float(row["init_cpu_cycles"])
                                  if "init_cpu_cycles" in row and row["init_cpu_cycles"]
                                  else 50000,
            )
            catalogue[svc.service_id] = svc
    return catalogue


def train(
    num_episodes   : int  = NUM_EPISODES,
    num_levels     : int  = NUM_LEVELS,
    num_children   : int  = NUM_CHILDREN,
    cache_cap      : int  = NODE_CACHE_CAP,
    num_services   : int  = NUM_SERVICES,
    req_per_ep     : int  = REQUESTS_PER_EPISODE,
    prov_interval  : int  = PROVISIONING_INTERVAL,
    seed           : int  = RANDOM_SEED,
    checkpoint_path: str  = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE),
    weights_path   : str  = os.path.join(CHECKPOINT_DIR, ONLINE_WEIGHTS_FILE),
    metrics_path   : str  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         RESULTS_DIR, METRICS_CSV),
    save_plots     : bool = True,
    traffic_model  : str  = TRAFFIC_MODEL,
    print_every    : int  = 10,
    services_csv   : str  = None,
    agent_type     : str  = "ddqn",
) -> EdgeOrchestrator:
    """
    Full training run.  Returns the EdgeOrchestrator so callers can
    inspect metrics or run post-training analysis.
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ?? 0. Seed ??????????????????????????????????????????????????????????
    _seed_all(seed)

    # ?? 1. Banner ????????????????????????????????????????????????????????
    print(_SEP)
    print(f"  Edge Caching  --  {agent_type.upper()} Training")
    print(f"  Started : {ts}")
    print(_SEP)
    print(f"  RL Agent      : {agent_type.upper()}")
    print(f"  Topology      : {num_levels} levels, {num_children} children, "
          f"cache_cap={cache_cap}")
    print(f"  Services      : {num_services}")
    print(f"  Episodes      : {num_episodes}")
    print(f"  Req / Episode : {req_per_ep}")
    print(f"  Prov Interval : {prov_interval}")
    print(f"  Traffic Model : {traffic_model}")
    print(f"  Seed          : {seed}")
    print(f"  Checkpoint    : {checkpoint_path}")
    print(_DIV)

    # ?? 2. Build world ????????????????????????????????????????????????????
    if services_csv and os.path.exists(services_csv):
        catalogue = _load_services_csv(services_csv)
        num_services = len(catalogue)
        print(f"  Services loaded  : {num_services} from {services_csv}")
    else:
        cat_list  = generate_service_catalogue(num_services, seed=seed)
        catalogue = {s.service_id: s for s in cat_list}

    topology = NetworkTopology(
        num_levels   = num_levels,
        num_children = num_children,
        cache_cap    = cache_cap,
    )
    topology.prime_cloud_cache(catalogue)
    init_report = topology.priority_init(catalogue)

    req_gen = RequestGenerator(
        catalogue    = catalogue,
        leaf_nodes   = topology.leaf_nodes,
        seed         = seed,
        model        = traffic_model,
    )
    orchestrator = EdgeOrchestrator(
        nodes      = topology.all_nodes,
        catalogue  = catalogue,
        agent_type = agent_type,
    )

    print("\n  Cache warm-start (priority_init):")
    init_report.print_report()
    print()

    # ?? 3. Episode loop ???????????????????????????????????????????????????
    print("  Training ...\n")

    for episode in range(num_episodes):

        # ?? Dynamic Load Selection ??????????????????????????????????
        try:
            from edge_caching.config import USE_CURRICULUM_LEARNING, CURRICULUM_SCHEDULE
            use_curriculum = USE_CURRICULUM_LEARNING
        except (ImportError, AttributeError):
            use_curriculum = False

        if use_curriculum:
            # Curriculum: start easy, gradually increase
            req_per_ep = REQUESTS_PER_EPISODE
            for ep_threshold in sorted(CURRICULUM_SCHEDULE.keys(), reverse=True):
                if episode >= ep_threshold:
                    req_per_ep = CURRICULUM_SCHEDULE[ep_threshold]
                    break

            if episode in CURRICULUM_SCHEDULE:
                print(f"\n  ?? Curriculum Phase: Increasing to {req_per_ep} requests\n")

        elif USE_VARIABLE_LOAD:
            # Variable: random sampling
            req_per_ep = np.random.choice(LOAD_DISTRIBUTION, p=LOAD_WEIGHTS)

        else:
            # Fixed: default from config
            req_per_ep = REQUESTS_PER_EPISODE

        requests = req_gen.generate(req_per_ep)
        total_prov_calls_per_ep = req_per_ep // PROVISIONING_INTERVAL

        ep_reward        = 0.0
        ep_executed      = 0
        ep_leaf_executed = 0   # Fix 4: requests served at origin leaf only
        ep_forwarded     = 0
        ep_violations    = 0
        ep_loss          = 0.0
        prov_calls       = 0

        for idx, req in enumerate(requests):

            # ?? Fast process: route request ??????????????????????????????
            origin_node = topology.node_map[req.origin_node_id]
            origin_node.receive_request(req, catalogue)

            if req.status == "executed":
                ep_executed += 1
                if req.executed_at_node == req.origin_node_id:
                    ep_leaf_executed += 1   # Fix 4: local-only hit
                else:
                    ep_forwarded += 1
            elif req.status == "dropped":
                ep_violations += 1

            # ?? Slow process: provision every N requests ?????????????????
            if (idx + 1) % PROVISIONING_INTERVAL == 0:
                prov_calls += 1
                is_last = (prov_calls == total_prov_calls_per_ep)
                metrics = orchestrator.provision(
                    episode_done = is_last,
                    sim_time_ms  = req_gen.sim_time,
                )
                ep_reward += metrics["interval_reward"]
                ep_loss   += metrics["avg_loss"]

        # ?? Popularity drift between episodes ????????????????????????????
        for svc in catalogue.values():
            svc.drift_popularity()

        avg_loss       = ep_loss / max(prov_calls, 1)
        # Fix 4: report local hit rate (leaf-only) to match throughput test metric.
        # ep_executed includes cloud-executed requests which inflate the rate;
        # ep_leaf_executed counts only requests served at the origin leaf node.
        hit_rate       = ep_executed      / max(req_per_ep, 1)   # total (incl. cloud)
        leaf_hit_rate  = ep_leaf_executed / max(req_per_ep, 1)   # local-only

        # ?? Episode boundary ?????????????????????????????????????????????
        orchestrator.end_episode(
            episode_reward   = ep_reward,
            total_executed   = ep_leaf_executed,   # Fix 4: use leaf-only count
            total_requests   = req_per_ep,
            total_forwarded  = ep_forwarded,
            total_violations = ep_violations,
            avg_loss         = avg_loss,
        )

        if (episode + 1) % print_every == 0 or episode == 0:
            _print_episode_summary(
                ep         = episode,
                total      = num_episodes,
                reward     = ep_reward,
                hit_rate   = leaf_hit_rate,        # Fix 4: print local rate
                violations = ep_violations,
                avg_loss   = avg_loss,
                epsilon    = orchestrator.agent.epsilon,
            )

    # ?? 4. Save checkpoint ????????????????????????????????????????????????
    print(f"\n{_DIV}")
    print("  Saving outputs ...")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    orchestrator.save_checkpoint(checkpoint_path)
    print(f"  Full checkpoint  -> {checkpoint_path}")

    # Lightweight online-weights-only export for the inference/simulation path
    torch.save(
        orchestrator.agent.online_net.state_dict(),
        weights_path,
    )
    print(f"  Online weights   -> {weights_path}")

    # ?? 5. Save metrics CSV ???????????????????????????????????????????????
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _save_metrics(orchestrator, metrics_path)
    print(f"  Training metrics -> {metrics_path}")

    # ?? 6. Save visualisation plots ???????????????????????????????????????
    if save_plots:
        save_all_results(orchestrator, topology)
        print(f"  Plots            -> {RESULTS_DIR}/")

    # ?? 7. Final summary ??????????????????????????????????????????????????
    final_hit   = orchestrator.episode_hit_rates[-1]  if orchestrator.episode_hit_rates  else 0.0
    final_loss  = orchestrator.episode_losses[-1]     if orchestrator.episode_losses      else 0.0
    final_viol  = orchestrator.episode_violations[-1] if orchestrator.episode_violations  else 0
    final_eps   = orchestrator.agent.epsilon

    print(_DIV)
    print("  Training complete")
    print(f"  Final local hit rate : {final_hit*100:.1f}%  (leaf-only, matches throughput test)")
    print(f"  Final avg loss       : {final_loss:.6f}")
    print(f"  Final violations     : {final_viol}")
    print(f"  Final epsilon        : {final_eps:.4f}")
    print(_SEP)

    return orchestrator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the DDQN edge-caching agent and save a model checkpoint."
    )
    p.add_argument("--episodes",    type=int,   default=NUM_EPISODES,
                   help=f"Number of training episodes (default: {NUM_EPISODES})")
    p.add_argument("--seed",        type=int,   default=RANDOM_SEED,
                   help=f"Random seed (default: {RANDOM_SEED})")
    p.add_argument("--checkpoint",  type=str,
                   default=os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE),
                   help="Path to save the full DDQN checkpoint (.pt)")
    p.add_argument("--weights",     type=str,
                   default=os.path.join(CHECKPOINT_DIR, ONLINE_WEIGHTS_FILE),
                   help="Path to save the online-network weights (.pt)")
    p.add_argument("--no-plots",    action="store_true",
                   help="Skip saving visualisation plots (useful for CI/CD runs)")
    p.add_argument("--print-every", type=int,   default=10,
                   help="Print a summary line every N episodes (default: 10)")
    p.add_argument("--traffic",     type=str,   default=TRAFFIC_MODEL,
                   choices=["poisson_zipf", "uniform"],
                   help="Traffic arrival model (default: poisson_zipf)")
    p.add_argument("--services",    type=str,   default=None,
                   help="Path to service catalogue CSV (default: auto-generated)")
    p.add_argument("--agent-type",  type=str,   default="ddqn",
                   choices=["ddqn", "dqn"],
                   help="RL algorithm to use: 'ddqn' (default) or 'dqn' (ablation baseline)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        num_episodes    = args.episodes,
        seed            = args.seed,
        checkpoint_path = args.checkpoint,
        weights_path    = args.weights,
        save_plots      = not args.no_plots,
        print_every     = args.print_every,
        traffic_model   = args.traffic,
        services_csv    = args.services,
        agent_type      = args.agent_type,
    )
