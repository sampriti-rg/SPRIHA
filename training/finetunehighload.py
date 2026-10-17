"""
finetunehighload.py
-------------------
Targeted high-load fine-tuning for an already fine-tuned DDQN checkpoint.

Purpose
-------
The existing fine-tuned model (ddqn_finetuned.pt) has a converged policy
for requests <= 7000. This script ONLY trains on load levels 7000-10000
to improve cache eviction decisions under sustained pressure WITHOUT
disturbing the low-load policy.

Key design decisions
--------------------
1. NO low-load episodes -- avoids overwriting converged low-load Q-values.
2. NO buffer pre-fill   -- buffer stays on-policy from the loaded checkpoint.
3. Low epsilon (0.05)   -- near-greedy so the agent refines rather than
                           re-explores. Existing good decisions are preserved.
4. total_prov recomputed each episode -- fixes the stale is_last bug.

Usage
-----
    python training/finetunehighload.py
    python training/finetunehighload.py --checkpoint checkpoints/ddqn_finetuned.pt
    python training/finetunehighload.py --episodes 150 --min-load 7000 --max-load 10000
"""

import argparse
import csv
import os
import random
import datetime
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge_caching.config import (
    NUM_LEVELS, NUM_CHILDREN, NODE_CACHE_CAP,
    PROVISIONING_INTERVAL, RANDOM_SEED,
    RESULTS_DIR, TRAFFIC_MODEL,
    EPSILON_MIN,
)
from edge_caching.network.topology import NetworkTopology
from edge_caching.utils.simulation.request_generator import RequestGenerator
from edge_caching.control.orchestrator import EdgeOrchestrator
from edge_caching.agents.ddqn_agent import DDQNAgent
from edge_caching.utils.visualiser import save_all_results
from training.collect_stats import load_sample_services_with_names

_ROOT               = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR            = os.path.join(_ROOT, "training", "data")
CHECKPOINT_DIR      = os.path.join(_ROOT, "checkpoints")
DEFAULT_INPUT_CKPT  = os.path.join(CHECKPOINT_DIR, "ddqn_finetuned.pt")
DEFAULT_OUTPUT_CKPT = os.path.join(CHECKPOINT_DIR, "ddqn_highload.pt")
DEFAULT_OUTPUT_W    = os.path.join(CHECKPOINT_DIR, "ddqn_highload_weights.pt")
SAMPLE_SERVICES_CSV = os.path.join(DATA_DIR, "sample_services.csv")
METRICS_CSV         = os.path.join(_ROOT, RESULTS_DIR, "finetune_highload_metrics.csv")

# High-load request levels sampled uniformly each episode
_HIGH_LOAD_LEVELS   = [20000, 15000, 30000, 32000]

_SEP = "=" * 72
_DIV = "-" * 72


def finetune_highload(
    services_csv    : str   = SAMPLE_SERVICES_CSV,
    input_ckpt      : str   = DEFAULT_INPUT_CKPT,
    output_ckpt     : str   = DEFAULT_OUTPUT_CKPT,
    output_weights  : str   = DEFAULT_OUTPUT_W,
    num_episodes    : int   = 150,
    prov_interval   : int   = PROVISIONING_INTERVAL,
    seed            : int   = RANDOM_SEED,
    traffic_model   : str   = TRAFFIC_MODEL,
    epsilon         : float = EPSILON_MIN,   # near-greedy: refine, don't re-explore
    min_load        : int   = 20000,
    max_load        : int   = 32000,
    save_plots      : bool  = True,
    print_every     : int   = 10,
) -> EdgeOrchestrator:
    """
    Fine-tune ONLY on high-load episodes to improve the 7k-10k policy
    without touching the converged low-load behaviour.

    Parameters
    ----------
    input_ckpt  : already fine-tuned checkpoint (ddqn_finetuned.pt)
    epsilon     : fixed exploration rate during this phase.
                  Default EPSILON_MIN (0.05) = near-greedy.
                  The existing policy is mostly preserved; only
                  high-load cache-pressure decisions are updated.
    min_load    : minimum requests per episode (default 7000)
    max_load    : maximum requests per episode (default 10000)
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Clamp load levels to [min_load, max_load]
    load_levels = [l for l in _HIGH_LOAD_LEVELS if min_load <= l <= max_load]
    if not load_levels:
        raise ValueError(
            f"No load levels in [{min_load}, {max_load}]. "
            f"Available: {_HIGH_LOAD_LEVELS}"
        )

    print(_SEP)
    print("  Edge Caching  --  High-Load Fine-Tuning")
    print(f"  Started       : {ts}")
    print(f"  Input ckpt    : {input_ckpt}")
    print(f"  Output ckpt   : {output_ckpt}")
    print(f"  Episodes      : {num_episodes}")
    print(f"  Load levels   : {load_levels}  (sampled uniformly per episode)")
    print(f"  Epsilon       : {epsilon:.4f}  (near-greedy -- low-load policy preserved)")
    print(f"  Buffer fill   : SKIPPED  (on-policy buffer from input checkpoint)")
    print(_DIV)

    # -- 1. Service catalogue ---------------------------------------------
    print("\n  Loading service catalogue ...")
    catalogue, _ = load_sample_services_with_names(services_csv)
    print()

    # -- 2. Topology ------------------------------------------------------
    topology = NetworkTopology(
        num_levels=NUM_LEVELS, num_children=NUM_CHILDREN, cache_cap=NODE_CACHE_CAP,
    )
    topology.prime_cloud_cache(catalogue)
    report = topology.priority_init(catalogue)
    print("  Cache warm-start:")
    report.print_report()
    print()

    # -- 3. Request generator + orchestrator ------------------------------
    req_gen = RequestGenerator(
        catalogue=catalogue, leaf_nodes=topology.leaf_nodes,
        seed=seed, model=traffic_model,
    )
    orchestrator = EdgeOrchestrator(nodes=topology.all_nodes, catalogue=catalogue)

    # -- 4. Load checkpoint -----------------------------------------------
    if not os.path.exists(input_ckpt):
        raise FileNotFoundError(
            f"Input checkpoint not found: {input_ckpt}\n"
            "Run  python training/finetune.py  first."
        )

    orchestrator.load_checkpoint(input_ckpt)
    print(f"  Loaded checkpoint : {input_ckpt}")
    print(f"  Checkpoint epsilon: {orchestrator.agent.epsilon:.4f}  "
          f"-> overriding to {epsilon:.4f} (high-load refinement phase)")

    # Override epsilon: near-greedy so the existing policy is preserved.
    # A high epsilon here would re-explore and degrade the low-load policy.
    orchestrator.agent.epsilon = epsilon
    orchestrator._disable_hard_sync = True
    print(f"  Target sync       : SOFT only (hard sync disabled -- prevents oscillation)")

    print()

    # -- 5. High-load episode loop ----------------------------------------
    print("  High-load fine-tuning ...\n")
    print(f"  {'Ep':>4}  {'Load':>6}  {'Reward':>10}  {'HitRate':>8}  "
          f"{'Dropped':>8}  {'Loss':>10}  {'Eps':>7}")
    print(_DIV)

    metrics_rows: List[dict] = []
    best_hit_rate  : float      = 0.0                                    # ADD
    best_episode   : int        = 0                                      # ADD
    best_ckpt_path : str        = output_ckpt.replace(".pt", "_best.pt") # ADD

    for episode in range(num_episodes):

        # Sample uniformly from high-load levels only
        ep_load    = random.choice(load_levels)

        # FIX: recompute total_prov per episode so is_last is always correct
        total_prov = ep_load // prov_interval

        requests      = req_gen.generate(ep_load)
        ep_reward     = 0.0
        ep_executed   = 0
        ep_forwarded  = 0
        ep_violations = 0
        ep_loss       = 0.0
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
                is_last = (prov_calls == total_prov)
                m = orchestrator.provision(
                    episode_done = is_last,
                    sim_time_ms  = req_gen.sim_time,
                )
                ep_reward += m["interval_reward"]
                ep_loss   += m["avg_loss"]

        for svc in catalogue.values():
            svc.drift_popularity()

        avg_loss = ep_loss     / max(prov_calls, 1)
        hit_rate = ep_executed / max(ep_load, 1)

        orchestrator.end_episode(
            episode_reward   = ep_reward,
            total_executed   = ep_executed,
            total_requests   = ep_load,
            total_forwarded  = ep_forwarded,
            total_violations = ep_violations,
            avg_loss         = avg_loss,
        )

        # -- Save best checkpoint whenever hit rate improves --------------
        is_best = hit_rate > best_hit_rate                                 # ADD
        if is_best:                                                        # ADD
            best_hit_rate = hit_rate                                       # ADD
            best_episode  = episode + 1                                    # ADD
            orchestrator.save_checkpoint(best_ckpt_path)

        metrics_rows.append({
            "episode"   : episode + 1,
            "load"      : ep_load,
            "reward"    : round(ep_reward, 4),
            "hit_rate"  : round(hit_rate,  4),
            "violations": ep_violations,
            "avg_loss"  : round(avg_loss,  6),
            "epsilon"   : round(orchestrator.agent.epsilon, 6),
            "is_best"   : int(is_best),
        })

        if (episode + 1) % print_every == 0 or episode == 0:
            print(
                f"  {episode+1:>4}  {ep_load:>6}  "
                f"{ep_reward:>10.3f}  "
                f"{hit_rate*100:>7.1f}%  "
                f"{ep_violations:>8}  "
                f"{avg_loss:>10.5f}  "
                f"{orchestrator.agent.epsilon:>7.4f}"
                f"{'  *' if is_best else '   '}"  
            )

    # -- 6. Save outputs --------------------------------------------------
    print(f"\n{_DIV}")
    print("  Saving outputs ...")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    orchestrator.save_checkpoint(output_ckpt)
    print(f"  Final checkpoint  -> {output_ckpt}  (ep {num_episodes})")
    print(f"  Best checkpoint   -> {best_ckpt_path}  "          # ADD
          f"(ep {best_episode}, hit {best_hit_rate*100:.1f}%)")  # ADD

    torch.save(orchestrator.agent.online_net.state_dict(), output_weights)
    print(f"  Weights        -> {output_weights}")

    os.makedirs(os.path.join(_ROOT, RESULTS_DIR), exist_ok=True)
    _save_metrics(metrics_rows, METRICS_CSV)
    print(f"  Metrics CSV    -> {METRICS_CSV}")

    if save_plots:
        save_all_results(orchestrator, topology)
        print(f"  Plots          -> {os.path.join(_ROOT, RESULTS_DIR)}/")

    final_hit  = orchestrator.episode_hit_rates[-1]  if orchestrator.episode_hit_rates  else 0.0
    final_loss = orchestrator.episode_losses[-1]     if orchestrator.episode_losses      else 0.0
    final_viol = orchestrator.episode_violations[-1] if orchestrator.episode_violations  else 0

    print(_DIV)
    print("  High-load fine-tuning complete")
    print(f"  Final hit rate   : {final_hit*100:.1f}%  (ep {num_episodes})")
    print(f"  Best hit rate    : {best_hit_rate*100:.1f}%  (ep {best_episode})")  # ADD
    print(f"  Final avg loss   : {final_loss:.6f}")
    print(f"  Final violations : {final_viol}")
    print(f"  Final epsilon    : {orchestrator.agent.epsilon:.4f}")
    print(_SEP)
    print()
    print("  Evaluate with best checkpoint:")                                      # ADD
    print(f"    python evaluate_ddqn_throughput.py --checkpoint {best_ckpt_path}") # ADD
    print()

    return orchestrator


def _save_metrics(rows: List[dict], path: str) -> None:
    fields = ["episode", "load", "reward", "hit_rate", "violations", "avg_loss", "epsilon", "is_best"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="High-load fine-tuning for an already fine-tuned DDQN checkpoint."
    )
    p.add_argument("--checkpoint",  type=str,   default=DEFAULT_INPUT_CKPT,
                   help=f"Fine-tuned input checkpoint (default: {DEFAULT_INPUT_CKPT})")
    p.add_argument("--output",      type=str,   default=DEFAULT_OUTPUT_CKPT,
                   help=f"Output checkpoint path (default: {DEFAULT_OUTPUT_CKPT})")
    p.add_argument("--services",    type=str,   default=SAMPLE_SERVICES_CSV,
                   help=f"Service catalogue CSV (default: {SAMPLE_SERVICES_CSV})")
    p.add_argument("--episodes",    type=int,   default=150,
                   help="Fine-tuning episodes (default: 150)")
    p.add_argument("--min-load",    type=int,   default=20000,
                   help="Minimum requests per episode (default: 7000)")
    p.add_argument("--max-load",    type=int,   default=35000,
                   help="Maximum requests per episode (default: 10000)")
    p.add_argument("--epsilon",     type=float, default=EPSILON_MIN,
                   help=f"Fixed epsilon during refinement (default: {EPSILON_MIN})")
    p.add_argument("--seed",        type=int,   default=RANDOM_SEED,
                   help=f"Random seed (default: {RANDOM_SEED})")
    p.add_argument("--traffic",     type=str,   default=TRAFFIC_MODEL,
                   choices=["poisson_zipf", "uniform"],
                   help="Traffic model (default: poisson_zipf)")
    p.add_argument("--no-plots",    action="store_true",
                   help="Skip saving visualisation plots")
    p.add_argument("--print-every", type=int,   default=10,
                   help="Print summary every N episodes (default: 10)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    finetune_highload(
        services_csv   = args.services,
        input_ckpt     = args.checkpoint,
        output_ckpt    = args.output,
        num_episodes   = args.episodes,
        prov_interval  = PROVISIONING_INTERVAL,
        seed           = args.seed,
        traffic_model  = args.traffic,
        epsilon        = args.epsilon,
        min_load       = args.min_load,
        max_load       = args.max_load,
        save_plots     = not args.no_plots,
        print_every    = args.print_every,
    )