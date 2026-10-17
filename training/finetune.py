"""
finetune.py
-----------
Phase 2 of the realistic-data training pipeline.

Purpose
-------
Load the realistic transitions collected by collect_stats.py, pre-populate
the DDQN replay buffer with them, then continue training from either:
  a) the existing checkpoint (checkpoints/ddqn_final.pt)  -- recommended
  b) a fresh agent initialised to the realistic service catalogue

This bridges the gap between the randomly-trained model and real-world
service demand patterns by:

  1. Reading collected_transitions.csv  (output of collect_stats.py)
  2. Pre-filling the replay buffer with those realistic (s, a, r, s', done)
     transitions so the first gradient updates see real demand structure.
  3. Running further training episodes using the SAME realistic catalogue
     (loaded from data/sample_services.csv) so popularity drift and EMA
     updates continue to reflect real service characteristics.
  4. Saving a new checkpoint: checkpoints/ddqn_finetuned.pt

Why this works
--------------
The DDQN policy is fully determined by the distribution of (s, a, r, s')
tuples in the replay buffer. By replacing the random transitions with
realistic ones, the agent's Q-function estimates are calibrated to the
true relative value of caching VideoStreaming vs BackupService, the true
cost of forwarding VoIP requests, and the true SLA violation rate under
realistic deadlines.

After fine-tuning the policy will:
  - Preferentially INSERT high-revenue, high-criticality services (VoIP,
    VideoStreaming) near the leaf level.
  - DELETE or KEEP low-popularity, low-revenue services (BackupService).
  - React correctly to cache pressure by evicting least-valuable services.

Usage
-----
    python finetune.py                          # fine-tune from existing checkpoint
    python finetune.py --fresh                  # start from scratch with real data
    python finetune.py --episodes 100           # number of fine-tuning episodes
    python finetune.py --transitions data/collected_transitions.csv
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

# Ensure the project root is on sys.path when run as  python training/finetune.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge_caching.config import (
    NUM_LEVELS, NUM_CHILDREN, NODE_CACHE_CAP, NUM_SERVICES,
    REQUESTS_PER_EPISODE, PROVISIONING_INTERVAL, RANDOM_SEED,
    RESULTS_DIR, TRAFFIC_MODEL, HOP_LATENCY_MS,
    EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
)

# Variable episode lengths to expose model to longer-term state evolution
EPISODE_LENGTHS = [10000, 20000, 30000]

from edge_caching.network.topology import NetworkTopology
from edge_caching.utils.simulation.request_generator import RequestGenerator
from edge_caching.control.orchestrator import EdgeOrchestrator
from edge_caching.agents.ddqn_agent import DDQNAgent
from edge_caching.utils.visualiser import save_all_results
from training.collect_stats import load_sample_services_with_names

_ROOT                = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR             = os.path.join(_ROOT, "training", "data")
CHECKPOINT_DIR       = os.path.join(_ROOT, "checkpoints")
PRETRAINED_CKPT      = os.path.join(CHECKPOINT_DIR, "ddqn_final.pt")
FINETUNED_CKPT       = os.path.join(CHECKPOINT_DIR, "ddqn_finetuned.pt")
FINETUNED_WEIGHTS    = os.path.join(CHECKPOINT_DIR, "ddqn_finetuned_weights.pt")
TRANSITIONS_CSV      = os.path.join(DATA_DIR, "collected_transitions.csv")
SAMPLE_SERVICES_CSV  = os.path.join(DATA_DIR, "sample_services.csv")
METRICS_CSV          = os.path.join(_ROOT, RESULTS_DIR, "finetune_metrics.csv")

_SEP = "=" * 72
_DIV = "-" * 72


# ---------------------------------------------------------------------------
# Load transitions from CSV into replay buffer
# ---------------------------------------------------------------------------

def load_transitions_into_buffer(
    agent           : DDQNAgent,
    transitions_csv : str,
    max_transitions : int = 5000,
) -> int:
    """
    Read (s, a, r, s', done) rows from CSV and push them into the agent's
    replay buffer using streaming reservoir sampling.

    The CSV is produced by collect_stats.py and has columns:
        s0..s7, action, reward, ns0..ns7, done

    Uses reservoir sampling to uniformly sample max_transitions rows from
    arbitrarily large CSV files without loading everything into memory.
    This prevents memory exhaustion and system freezes on multi-GB datasets.

    Parameters
    ----------
    agent           : DDQNAgent whose memory buffer will be filled
    transitions_csv : path to collected_transitions.csv
    max_transitions : cap on how many rows to load (default = MEMORY_SIZE)

    Returns
    -------
    n_loaded : int  number of transitions pushed into the buffer
    """
    if not os.path.exists(transitions_csv):
        raise FileNotFoundError(
            f"Transitions file not found: {transitions_csv}\n"
            "Run  python collect_stats.py  first."
        )

    print(f"  Streaming transitions from {os.path.basename(transitions_csv)} ...")
    print(f"  Using reservoir sampling (max {max_transitions:,} transitions)")

    reservoir: List[dict] = []
    state_dim = 10  # Default, will be detected from first row

    with open(transitions_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for idx, row in enumerate(reader):
            # Detect state dimension from first row
            if idx == 0:
                state_dim = sum(1 for k in row if k.startswith("s") and k[1:].isdigit())
            
            # Reservoir sampling: fill first k items, then probabilistically replace
            if idx < max_transitions:
                reservoir.append(row)
            else:
                # For each item after the first k, randomly decide whether to include it
                # with probability k/idx to maintain uniform distribution
                j = random.randint(0, idx)
                if j < max_transitions:
                    reservoir[j] = row
            
            # Progress indicator for large files
            if (idx + 1) % 100000 == 0:
                print(f"    Processed {idx + 1:,} rows, reservoir size: {len(reservoir):,}")

    print(f"  Loaded {len(reservoir):,} transitions from {idx + 1:,} total rows")

    # Shuffle reservoir to break any remaining temporal correlation
    random.shuffle(reservoir)

    # Push sampled transitions into replay buffer
    for row in reservoir:
        state      = np.array([float(row[f"s{i}"]) for i in range(state_dim)], dtype=np.float32)
        next_state = np.array([float(row[f"ns{i}"]) for i in range(state_dim)], dtype=np.float32)
        action     = int(row["action"])
        reward     = float(row["reward"])
        done       = bool(int(row["done"]))
        agent.memory.push(state, action, reward, next_state, done)

    return len(reservoir)


# ---------------------------------------------------------------------------
# Fine-tuning loop
# ---------------------------------------------------------------------------

def finetune(
    transitions_csv : str  = TRANSITIONS_CSV,
    services_csv    : str  = SAMPLE_SERVICES_CSV,
    pretrained_ckpt : str  = PRETRAINED_CKPT,
    output_ckpt     : str  = FINETUNED_CKPT,
    output_weights  : str  = FINETUNED_WEIGHTS,
    num_episodes    : int  = 100,
    req_per_ep      : int  = REQUESTS_PER_EPISODE,
    prov_interval   : int  = PROVISIONING_INTERVAL,
    seed            : int  = RANDOM_SEED,
    traffic_model   : str  = TRAFFIC_MODEL,
    start_fresh     : bool = False,
    save_plots      : bool = True,
    print_every     : int  = 10,
    prefill_buffer  : bool = True,   # <-- ADD THIS
) -> EdgeOrchestrator:
    """
    Fine-tune the DDQN agent using realistic collected transitions.

    Steps
    -----
    1. Load realistic service catalogue from sample_services.csv
    2. Build topology and orchestrator
    3. Optionally load pre-trained weights from pretrained_ckpt
    4. Pre-fill replay buffer from collected_transitions.csv
    5. Run num_episodes of training with realistic services
    6. Save fine-tuned checkpoint
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(_SEP)
    print("  Edge Caching  --  DDQN Fine-Tuning (Realistic Data)")
    print(f"  Started          : {ts}")
    print(f"  Transitions CSV  : {transitions_csv}")
    print(f"  Services CSV     : {services_csv}")
    print(f"  Pre-trained ckpt : {pretrained_ckpt}  ({'SKIP - fresh start' if start_fresh else 'LOAD'})")
    print(f"  Output ckpt      : {output_ckpt}")
    print(f"  Episodes         : {num_episodes}")
    print(_DIV)

    # ?? 1. Load realistic catalogue ???????????????????????????????????????
    print("\n  Loading realistic service catalogue ...")
    catalogue, names = load_sample_services_with_names(services_csv)
    print()

    # ?? 2. Build topology ?????????????????????????????????????????????????
    topology = NetworkTopology(
        num_levels=NUM_LEVELS, num_children=NUM_CHILDREN, cache_cap=NODE_CACHE_CAP,
    )
    topology.prime_cloud_cache(catalogue)
    report = topology.priority_init(catalogue)

    print("  Cache warm-start (priority_init):")
    report.print_report()
    print()

    # ?? 3. Orchestrator ???????????????????????????????????????????????????
    req_gen = RequestGenerator(
        catalogue=catalogue, leaf_nodes=topology.leaf_nodes,
        seed=seed, model=traffic_model,
    )
    orchestrator = EdgeOrchestrator(nodes=topology.all_nodes, catalogue=catalogue)

    # ?? 4. Load pre-trained weights or start fresh ????????????????????????
    if not start_fresh and os.path.exists(pretrained_ckpt):
        try:
            orchestrator.load_checkpoint(pretrained_ckpt)
            print(f"  Loaded pre-trained checkpoint: {pretrained_ckpt}")
            print(f"  Resuming from epsilon = {orchestrator.agent.epsilon:.4f}")
        except RuntimeError as e:
            print(f"  WARNING: checkpoint incompatible ({e})")
            print("  Falling back to fresh weights (state-dim mismatch - retrain needed).")
            orchestrator.agent.epsilon = EPSILON_START
    else:
        print("  Starting from freshly initialised weights.")
        orchestrator.agent.epsilon = EPSILON_START

    # ?? 5. Pre-fill replay buffer with realistic transitions ??????????????
    if prefill_buffer:
        print(f"\n  Pre-filling replay buffer from {transitions_csv} ...")
        n_loaded = load_transitions_into_buffer(
            agent           = orchestrator.agent,
            transitions_csv = transitions_csv,
            max_transitions = orchestrator.agent.memory.capacity,
        )
        buffer_fill_pct = 100 * n_loaded / orchestrator.agent.memory.capacity
        print(f"  Loaded {n_loaded} transitions  "
              f"({buffer_fill_pct:.1f}% of buffer capacity {orchestrator.agent.memory.capacity})")
        print(f"  Buffer is_ready() : {orchestrator.agent.memory.is_ready()}")
        print()

    # ?? 6. Fine-tuning episode loop ???????????????????????????????????????
    print("  Fine-tuning ...\n")
    print(f"  Variable episode lengths: {EPISODE_LENGTHS}\n")
    metrics_rows: List[dict] = []

    for episode in range(num_episodes):
        # Randomly select episode length for this episode
        requests_this_episode = random.choice(EPISODE_LENGTHS)
        total_prov = requests_this_episode // prov_interval

        requests      = req_gen.generate(requests_this_episode)
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

        avg_loss = ep_loss / max(prov_calls, 1)
        hit_rate = ep_executed / max(requests_this_episode, 1)

        orchestrator.end_episode(
            episode_reward=ep_reward, total_executed=ep_executed,
            total_requests=requests_this_episode, total_forwarded=ep_forwarded,
            total_violations=ep_violations, avg_loss=avg_loss,
        )

        metrics_rows.append({
            "episode"   : episode + 1,
            "reward"    : round(ep_reward, 4),
            "hit_rate"  : round(hit_rate,  4),
            "violations": ep_violations,
            "avg_loss"  : round(avg_loss,  6),
            "epsilon"   : round(orchestrator.agent.epsilon, 6),
            "episode_length": requests_this_episode,
        })

        if (episode + 1) % print_every == 0 or episode == 0:
            print(
                f"  Ep {episode+1:>4}/{num_episodes} (Len={requests_this_episode:,})  |  "
                f"Reward: {ep_reward:>8.3f}  |  "
                f"HitRate: {hit_rate*100:>5.1f}%  |  "
                f"Violations: {ep_violations:>4}  |  "
                f"Loss: {avg_loss:>8.5f}  |  "
                f"e: {orchestrator.agent.epsilon:.4f}"
            )

    # ?? 7. Save outputs ???????????????????????????????????????????????????
    print(f"\n{_DIV}")
    print("  Saving outputs ...")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    orchestrator.save_checkpoint(output_ckpt)
    print(f"  Fine-tuned checkpoint -> {output_ckpt}")

    torch.save(orchestrator.agent.online_net.state_dict(), output_weights)
    print(f"  Fine-tuned weights    -> {output_weights}")

    os.makedirs(os.path.join(_ROOT, RESULTS_DIR), exist_ok=True)
    _save_metrics(metrics_rows, METRICS_CSV)
    print(f"  Fine-tune metrics     -> {METRICS_CSV}")

    if save_plots:
        save_all_results(orchestrator, topology)
        print(f"  Plots                 -> {os.path.join(_ROOT, RESULTS_DIR)}/")

    # ?? 8. Final summary ??????????????????????????????????????????????????
    final_hit  = orchestrator.episode_hit_rates[-1]  if orchestrator.episode_hit_rates  else 0.0
    final_loss = orchestrator.episode_losses[-1]     if orchestrator.episode_losses      else 0.0
    final_viol = orchestrator.episode_violations[-1] if orchestrator.episode_violations  else 0

    print(_DIV)
    print("  Fine-tuning complete")
    print(f"  Final hit rate   : {final_hit*100:.1f}%")
    print(f"  Final avg loss   : {final_loss:.6f}")
    print(f"  Final violations : {final_viol}")
    print(f"  Final epsilon    : {orchestrator.agent.epsilon:.4f}")
    print(_SEP)
    print()
    print("  Use the fine-tuned model for inference:")
    print(f"    python inference.py --checkpoint {output_ckpt}")
    print()

    return orchestrator


def _save_metrics(rows: List[dict], path: str) -> None:
    fields = ["episode", "reward", "hit_rate", "violations", "avg_loss", "epsilon", "episode_length"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune DDQN with realistic collected statistics."
    )
    p.add_argument("--transitions", type=str, default=TRANSITIONS_CSV,
                   help=f"Path to collected_transitions.csv (default: {TRANSITIONS_CSV})")
    p.add_argument("--services",    type=str, default=SAMPLE_SERVICES_CSV,
                   help=f"Path to sample_services.csv (default: {SAMPLE_SERVICES_CSV})")
    p.add_argument("--checkpoint",  type=str, default=PRETRAINED_CKPT,
                   help=f"Pre-trained checkpoint to load (default: {PRETRAINED_CKPT})")
    p.add_argument("--output",      type=str, default=FINETUNED_CKPT,
                   help=f"Output checkpoint path (default: {FINETUNED_CKPT})")
    p.add_argument("--episodes",    type=int, default=100,
                   help="Number of fine-tuning episodes (default: 100)")
    p.add_argument("--seed",        type=int, default=RANDOM_SEED,
                   help=f"Random seed (default: {RANDOM_SEED})")
    p.add_argument("--traffic",     type=str, default=TRAFFIC_MODEL,
                   choices=["poisson_zipf", "uniform"],
                   help="Traffic model (default: poisson_zipf)")
    p.add_argument("--fresh",       action="store_true",
                   help="Start from fresh weights instead of loading pre-trained checkpoint")
    p.add_argument("--no-plots",    action="store_true",
                   help="Skip saving visualisation plots")
    p.add_argument("--print-every", type=int, default=10,
                   help="Print summary every N episodes (default: 10)")
    p.add_argument("--no-prefill",  action="store_true",
                   help="Skip pre-filling the replay buffer with realistic transitions")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    finetune(
        transitions_csv = args.transitions,
        services_csv    = args.services,
        pretrained_ckpt = args.checkpoint,
        output_ckpt     = args.output,
        num_episodes    = args.episodes,
        seed            = args.seed,
        traffic_model   = args.traffic,
        start_fresh     = args.fresh,
        save_plots      = not args.no_plots,
        print_every     = args.print_every,
        prefill_buffer  = not args.no_prefill,
    )
