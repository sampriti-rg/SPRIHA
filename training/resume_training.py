"""
Resume DDQN training from an existing checkpoint with proper metric tracking.
"""

import argparse
import csv
import os
import sys
import datetime
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from edge_caching.config import (
    NUM_LEVELS, NUM_CHILDREN, NODE_CACHE_CAP,
    REQUESTS_PER_EPISODE, PROVISIONING_INTERVAL,
    RANDOM_SEED, RESULTS_DIR, TRAFFIC_MODEL,
)

# Variable episode lengths to expose model to longer-term state evolution
EPISODE_LENGTHS = [20000,30000]
from edge_caching.core.service import generate_service_catalogue
from edge_caching.network.topology import NetworkTopology
from edge_caching.utils.simulation.request_generator import RequestGenerator
from edge_caching.control.orchestrator import EdgeOrchestrator
from edge_caching.utils.visualiser import save_all_results
from training.train import _seed_all, _load_services_csv, _print_episode_summary


def load_existing_metrics(path: str):
    """Load existing training metrics to determine starting episode."""
    if not os.path.exists(path):
        return [], 0
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    last_episode = int(rows[-1]['episode']) if rows else 0
    return rows, last_episode


def save_merged_metrics(existing_rows, new_metrics, path: str, episode_offset: int):
    """Merge and save metrics with corrected episode numbers."""
    
    fieldnames = [
        "episode", "reward", "profit", "hit_rate",
        "forwarded", "violations", "avg_loss", "epsilon", "episode_length"
    ]
    
    # Adjust new episode numbers
    adjusted_new = []
    for ep_idx, metrics in enumerate(new_metrics):
        adjusted_new.append({
            "episode": episode_offset + ep_idx + 1,
            **metrics
        })
    
    # Write combined data
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows + adjusted_new)
    
    print(f"Metrics updated: {len(existing_rows)} + {len(adjusted_new)} = {len(existing_rows) + len(adjusted_new)} episodes")


def resume_training():
    parser = argparse.ArgumentParser(description="Resume DDQN training from checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--episodes", type=int, default=200, help="Additional episodes to train")
    parser.add_argument("--services", default="training/data/sample_services.csv")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--noplots", action="store_true")
    
    args = parser.parse_args()
    
    # Load existing metrics to determine offset
    metrics_path = os.path.join(RESULTS_DIR, "training_metrics.csv")
    existing_metrics, episode_offset = load_existing_metrics(metrics_path)
    
    print("=" * 72)
    print("  RESUMING DDQN TRAINING")
    print("=" * 72)
    print(f"  Checkpoint:       {args.checkpoint}")
    print(f"  Previous episodes: {episode_offset}")
    print(f"  New episodes:     {episode_offset + 1} :  {episode_offset + args.episodes}")
    print(f"  Additional:       {args.episodes} episodes")
    print("=" * 72)
    
    _seed_all(RANDOM_SEED)
    
    # Build world
    if args.services and os.path.exists(args.services):
        catalogue = _load_services_csv(args.services)
        num_services = len(catalogue)
        print(f"  Services loaded:  {num_services} from {args.services}")
    else:
        cat_list = generate_service_catalogue(50, seed=RANDOM_SEED)
        catalogue = {s.service_id: s for s in cat_list}
    
    topology = NetworkTopology(NUM_LEVELS, NUM_CHILDREN, NODE_CACHE_CAP)
    topology.prime_cloud_cache(catalogue)
    topology.priority_init(catalogue)
    
    req_gen = RequestGenerator(
        catalogue=catalogue,
        leaf_nodes=topology.leaf_nodes,
        seed=RANDOM_SEED,
        model=TRAFFIC_MODEL,
    )
    
    orchestrator = EdgeOrchestrator(
        nodes=topology.all_nodes,
        catalogue=catalogue,
    )
    
    # Load checkpoint
    print(f"\n  Loading checkpoint: {args.checkpoint}")
    orchestrator.load_checkpoint(args.checkpoint)
    print(f"  Epsilon: {orchestrator.agent.epsilon:.6f}")
    print(f"  Replay buffer: {len(orchestrator.agent.memory)} transitions")
    
    # Warm up replay buffer if empty or too small
    min_buffer_size = 1000  # Minimum transitions needed for learning
    if len(orchestrator.agent.memory) < min_buffer_size:
        print(f"\n  ⚠ WARNING: Replay buffer has only {len(orchestrator.agent.memory)} transitions!")
        print(f"  Warming up buffer to at least {min_buffer_size} transitions...")
        
        # Save original epsilon
        original_epsilon = orchestrator.agent.epsilon
        orchestrator.agent.epsilon = 0.3  # Use moderate exploration during warmup
        
        warmup_requests_total = 5000  # Generate enough requests for ~100 provisioning calls
        warmup_requests = req_gen.generate(warmup_requests_total)
        
        warmup_count = 0
        for idx, req in enumerate(warmup_requests):
            origin_node = topology.node_map[req.origin_node_id]
            origin_node.receive_request(req, catalogue)
            
            if (idx + 1) % PROVISIONING_INTERVAL == 0:
                warmup_count += 1
                orchestrator.provision(
                    episode_done=False,
                    sim_time_ms=req_gen.sim_time,
                )
                
                # Stop when we have enough transitions
                if len(orchestrator.agent.memory) >= min_buffer_size:
                    break
        
        # Restore original epsilon
        orchestrator.agent.epsilon = original_epsilon
        
        print(f"  ✓ Buffer warmed up: {len(orchestrator.agent.memory)} transitions")
        print(f"  ✓ Provisioning calls during warmup: {warmup_count}")
        print(f"  ✓ Restored epsilon to: {orchestrator.agent.epsilon:.6f}\n")
    else:
        print(f"  ✓ Buffer has sufficient transitions for training\n")
    
    # Training loop
    print("  Resuming training ...\n")
    print(f"  Variable episode lengths: {EPISODE_LENGTHS}\n")
    
    new_episode_metrics = []
    
    for episode in range(args.episodes):
        # Randomly select episode length for this episode
        requests_this_episode = random.choice(EPISODE_LENGTHS)
        total_prov_calls_per_ep = requests_this_episode // PROVISIONING_INTERVAL
        
        requests = req_gen.generate(requests_this_episode)
        
        ep_reward = 0.0
        ep_executed = 0
        ep_leaf_executed = 0
        ep_forwarded = 0
        ep_violations = 0
        ep_loss = 0.0
        prov_calls = 0
        
        for idx, req in enumerate(requests):
            origin_node = topology.node_map[req.origin_node_id]
            origin_node.receive_request(req, catalogue)
            
            if req.status == "executed":
                ep_executed += 1
                if req.executed_at_node == req.origin_node_id:
                    ep_leaf_executed += 1
                else:
                    ep_forwarded += 1
            elif req.status == "dropped":
                ep_violations += 1
            
            if (idx + 1) % PROVISIONING_INTERVAL == 0:
                prov_calls += 1
                is_last = (prov_calls == total_prov_calls_per_ep)
                metrics = orchestrator.provision(
                    episode_done=is_last,
                    sim_time_ms=req_gen.sim_time,
                )
                ep_reward += metrics["interval_reward"]
                ep_loss += metrics["avg_loss"]
        
        # Popularity drift
        for svc in catalogue.values():
            svc.drift_popularity()
        
        avg_loss = ep_loss / max(prov_calls, 1)
        hit_rate = ep_executed / max(requests_this_episode, 1)
        leaf_hit_rate = ep_leaf_executed / max(requests_this_episode, 1)
        
        # Episode boundary
        orchestrator.end_episode(
            episode_reward=ep_reward,
            total_executed=ep_leaf_executed,
            total_requests=requests_this_episode,
            total_forwarded=ep_forwarded,
            total_violations=ep_violations,
            avg_loss=avg_loss,
        )
        
        # Store metrics
        new_episode_metrics.append({
            "reward": round(ep_reward, 4),
            "profit": round(sum(n.total_revenue for n in topology.all_nodes), 4),
            "hit_rate": round(leaf_hit_rate, 4),
            "forwarded": ep_forwarded,
            "violations": ep_violations,
            "avg_loss": round(avg_loss, 6),
            "epsilon": round(orchestrator.agent.epsilon, 6),
            "episode_length": requests_this_episode,
        })
        
        if (episode + 1) % args.print_every == 0 or episode == 0:
            print(f"  [Ep {episode + 1}/{args.episodes}] Len={requests_this_episode:,} | ", end="")
            _print_episode_summary(
                ep=episode,
                total=args.episodes,
                reward=ep_reward,
                hit_rate=leaf_hit_rate,
                violations=ep_violations,
                avg_loss=avg_loss,
                epsilon=orchestrator.agent.epsilon,
            )
    
    # Save outputs
    print(f"\n{'=' * 72}")
    print("  Saving outputs ...")
    
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    final_checkpoint = os.path.join(checkpoint_dir, f"ddqn_resumed_{episode_offset + args.episodes}ep.pt")
    orchestrator.save_checkpoint(final_checkpoint)
    print(f"  Checkpoint: {final_checkpoint}")
    
    weights_path = os.path.join(checkpoint_dir, "ddqn_online_weights.pt")
    torch.save(orchestrator.agent.online_net.state_dict(), weights_path)
    print(f"  Online weights:{weights_path}")
    
    # Merge metrics
    save_merged_metrics(existing_metrics, new_episode_metrics, metrics_path, episode_offset)
    print(f"  Training metrics:{metrics_path}")
    
    # Plots
    if not args.no_plots:
        save_all_results(orchestrator, topology)
        print(f"  Plots:{RESULTS_DIR}/")
    
    print("=" * 72)
    print(f"  TRAINING COMPLETE!")
    print(f"  Total episodes: {episode_offset + args.episodes}")
    print(f"  Final epsilon: {orchestrator.agent.epsilon:.6f}")
    print("=" * 72)


if __name__ == "__main__":
    resume_training()