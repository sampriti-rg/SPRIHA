"""
Performance Profiler for DDQN Training
---------------------------------------
Profiles 1-2 episodes to identify bottlenecks and optimization opportunities.

Usage:
    python profile_training.py
    python profile_training.py --episodes 2 --requests 5000
"""

import argparse
import cProfile
import pstats
import io
import sys
import os
import time
import json
from datetime import datetime
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edge_caching.config import (
    NUM_LEVELS, NUM_CHILDREN, NODE_CACHE_CAP, NUM_SERVICES,
    REQUESTS_PER_EPISODE, PROVISIONING_INTERVAL, RANDOM_SEED
)
from edge_caching.core.service import generate_service_catalogue
from edge_caching.network.topology import NetworkTopology
from edge_caching.utils.simulation.request_generator import RequestGenerator
from edge_caching.control.orchestrator import EdgeOrchestrator


class PerformanceProfiler:
    """Tracks timing and performance metrics."""
    
    def __init__(self):
        self.timings = {}
    
    def record(self, name, duration):
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration)
    
    def summary(self):
        """Get summary statistics."""
        summary = {}
        for name, times in self.timings.items():
            summary[name] = {
                'count': len(times),
                'total_ms': sum(times) * 1000,
                'mean_ms': np.mean(times) * 1000,
                'min_ms': np.min(times) * 1000,
                'max_ms': np.max(times) * 1000,
            }
        return summary


def run_profiled_training(num_episodes=2, req_per_ep=REQUESTS_PER_EPISODE):
    """Run training with detailed profiling."""
    
    profiler = PerformanceProfiler()
    
    print("=" * 72)
    print("PROFILED TRAINING SESSION")
    print("-" * 72)
    print(f"Episodes: {num_episodes}")
    print(f"Requests/Episode: {req_per_ep}")
    print(f"Provisioning Interval: {PROVISIONING_INTERVAL}")
    print("=" * 72)
    
    # Setup
    print("\nSetting up simulation...")
    setup_start = time.perf_counter()
    
    catalogue_list = generate_service_catalogue(NUM_SERVICES, seed=RANDOM_SEED)
    catalogue = {s.service_id: s for s in catalogue_list}
    
    topology = NetworkTopology(
        num_levels=NUM_LEVELS,
        num_children=NUM_CHILDREN,
        cache_cap=NODE_CACHE_CAP,
    )
    topology.prime_cloud_cache(catalogue)
    topology.priority_init(catalogue)
    
    req_gen = RequestGenerator(
        catalogue=catalogue,
        leaf_nodes=topology.leaf_nodes,
        seed=RANDOM_SEED,
    )
    
    orchestrator = EdgeOrchestrator(
        nodes=topology.all_nodes,
        catalogue=catalogue,
    )
    
    setup_time = time.perf_counter() - setup_start
    print(f"Setup complete ({setup_time:.2f}s)")
    print(f"  Nodes: {len(topology.all_nodes)}")
    print(f"  Services: {NUM_SERVICES}")
    
    # Training loop
    episode_times = []
    
    for episode in range(num_episodes):
        print(f"\n{'='*72}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print('-'*72)
        episode_start = time.perf_counter()
        
        # Generate requests
        gen_start = time.perf_counter()
        requests = req_gen.generate(req_per_ep)
        gen_time = time.perf_counter() - gen_start
        profiler.record('request_generation', gen_time)
        print(f"Request generation: {gen_time*1000:.2f}ms")
        
        ep_reward = 0.0
        ep_executed = 0
        ep_forwarded = 0
        ep_violations = 0
        ep_loss = 0.0
        prov_calls = 0
        
        total_prov_calls = req_per_ep // PROVISIONING_INTERVAL
        
        routing_times = []
        provision_times = []
        
        # Process requests
        for idx, req in enumerate(requests):
            # Fast process: routing
            route_start = time.perf_counter()
            origin_node = topology.node_map[req.origin_node_id]
            origin_node.receive_request(req, catalogue)
            routing_times.append(time.perf_counter() - route_start)
            
            # Count outcomes
            if req.status == "executed":
                ep_executed += 1
                if req.executed_at_node != req.origin_node_id:
                    ep_forwarded += 1
            elif req.status == "dropped":
                ep_violations += 1
            
            # Slow process: provisioning
            if (idx + 1) % PROVISIONING_INTERVAL == 0:
                prov_start = time.perf_counter()
                prov_calls += 1
                is_last = (prov_calls == total_prov_calls)
                
                metrics = orchestrator.provision(
                    episode_done=is_last,
                    sim_time_ms=req_gen.sim_time,
                )
                
                provision_times.append(time.perf_counter() - prov_start)
                ep_reward += metrics["interval_reward"]
                ep_loss += metrics["avg_loss"]
        
        # Record timing stats
        total_route = sum(routing_times)
        total_prov = sum(provision_times)
        profiler.record('total_routing', total_route)
        profiler.record('total_provisioning', total_prov)
        profiler.record('avg_routing_per_request', np.mean(routing_times))
        profiler.record('avg_provisioning_call', np.mean(provision_times))
        
        print(f"Routing: {total_route*1000:.2f}ms total ({np.mean(routing_times)*1000:.4f}ms avg)")
        print(f"Provisioning: {total_prov*1000:.2f}ms total ({np.mean(provision_times)*1000:.2f}ms avg)")
        print(f"Provision calls: {prov_calls}")
        
        # Popularity drift
        for svc in catalogue.values():
            svc.drift_popularity()
        
        # End episode
        avg_loss = ep_loss / max(prov_calls, 1)
        orchestrator.end_episode(
            episode_reward=ep_reward,
            total_executed=ep_executed,
            total_requests=req_per_ep,
            total_forwarded=ep_forwarded,
            total_violations=ep_violations,
            avg_loss=avg_loss,
        )
        
        episode_time = time.perf_counter() - episode_start
        episode_times.append(episode_time)
        
        hit_rate = ep_executed / req_per_ep * 100
        print(f"Episode time: {episode_time:.3f}s")
        print(f"Reward: {ep_reward:.2f}, Hit rate: {hit_rate:.1f}%")
    
    print("\n" + "=" * 72)
    print("PROFILING COMPLETE")
    print("-" * 72)
    print(f"Total time: {sum(episode_times):.3f}s")
    print(f"Avg episode time: {np.mean(episode_times):.3f}s")
    print("=" * 72)
    
    return {
        'episode_times': episode_times,
        'avg_episode_time': np.mean(episode_times),
        'timing_stats': profiler.summary(),
    }


def analyze_and_report(results):
    """Analyze bottlenecks and generate recommendations."""
    
    print("\n" + "=" * 72)
    print("BOTTLENECK ANALYSIS")
    print("=" * 72)
    
    stats = results['timing_stats']
    items = [(name, data) for name, data in stats.items()]
    items.sort(key=lambda x: x[1]['total_ms'], reverse=True)
    
    print("\nTime Distribution:")
    print("-" * 72)
    print(f"{'Component':<35} {'Total (ms)':<12} {'Count':<8} {'Avg (ms)':<10}")
    print("-" * 72)
    
    for name, data in items:
        print(f"{name:<35} {data['total_ms']:>10.2f}  {data['count']:>6}  {data['mean_ms']:>8.4f}")
    
    print("\n" + "=" * 72)
    print("RECOMMENDATIONS")
    print("=" * 72)
    
    recommendations = []
    
    # Analyze provisioning
    prov_stats = [s for n, s in items if 'provisioning' in n.lower()]
    if prov_stats and prov_stats[0]['total_ms'] > 1000:
        recommendations.append({
            'priority': 'HIGH',
            'component': 'Provisioning (DDQN)',
            'issue': f"Takes {prov_stats[0]['total_ms']/1000:.2f}s total",
            'solutions': [
                '1. Use agent.select_actions_batch() for vectorized action selection',
                '2. Increase PROVISIONING_INTERVAL to reduce frequency',
                '3. Reduce state vector computation overhead',
            ]
        })
    
    # Analyze routing
    route_stats = [s for n, s in items if 'routing' in n.lower()]
    if route_stats and route_stats[0]['total_ms'] > 500:
        recommendations.append({
            'priority': 'MEDIUM',
            'component': 'Request Routing',
            'issue': f"Takes {route_stats[0]['total_ms']/1000:.2f}s total",
            'solutions': [
                '1. Cache topology node lookups',
                '2. Optimize EDF queue operations',
                '3. Profile node.receive_request() method',
            ]
        })
    
    # Check GPU
    if not torch.cuda.is_available():
        recommendations.append({
            'priority': 'CRITICAL',
            'component': 'GPU Acceleration',
            'issue': 'GPU not available - training on CPU',
            'solutions': [
                '1. Install CUDA-enabled PyTorch for 6-8x speedup',
                '2. Run: pip install torch --index-url https://pytorch.org/whl/cu118',
                '3. Verify GPU: nvidia-smi',
            ]
        })
    
    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. [{rec['priority']}] {rec['component']}")
        print(f"   Issue: {rec['issue']}")
        print("   Solutions:")
        for sol in rec['solutions']:
            print(f"      {sol}")
    
    if not recommendations:
        print("\n No major bottlenecks detected - performance is well optimized!")
    
    return recommendations


def save_results(results, recommendations, output_dir='profiling_results'):
    """Save profiling results to JSON."""
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = {
        'timestamp': timestamp,
        'summary': {
            'total_time_s': sum(results['episode_times']),
            'avg_episode_time_s': results['avg_episode_time'],
            'gpu_available': torch.cuda.is_available(),
        },
        'timing_stats': results['timing_stats'],
        'recommendations': recommendations,
    }
    
    filepath = os.path.join(output_dir, f"profile_{timestamp}.json")
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n Results saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description='Profile DDQN training performance for bottleneck analysis'
    )
    parser.add_argument(
        '--episodes', type=int, default=2,
        help='Number of episodes to profile (default: 2)'
    )
    parser.add_argument(
        '--requests', type=int, default=REQUESTS_PER_EPISODE,
        help=f'Requests per episode (default: {REQUESTS_PER_EPISODE})'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 72)
    print("DDQN TRAINING PERFORMANCE PROFILER")
    print("=" * 72)
    print(f"Configuration:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Requests/Episode: {args.requests}")
    print(f"  GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 72)
    
    # Run profiling with cProfile
    print("\nRunning cProfile...")
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run training
    results = run_profiled_training(args.episodes, args.requests)
    
    profiler.disable()
    
    # Print cProfile top functions
    print("\n" + "=" * 72)
    print("CPROFILE - TOP 25 FUNCTIONS BY CUMULATIVE TIME")
    print("=" * 72)
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(25)
    print(s.getvalue())
    
    # Analyze bottlenecks
    recommendations = analyze_and_report(results)
    
    # Save results
    save_results(results, recommendations)
    
    # Save cProfile stats
    os.makedirs('profiling_results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_file = f"profiling_results/cprofile_{timestamp}.prof"
    stats.dump_stats(stats_file)
    print(f" cProfile stats saved to: {stats_file}")
    print(f"  View with: python -m pstats {stats_file}")
    
    print("\n" + "=" * 72)
    print("PROFILING COMPLETE")
    print("=" * 72)
    print("\nNext Steps:")
    print("1. Review bottleneck analysis above")
    print("2. Implement high-priority recommendations")
    print("3. Re-run profiler to measure improvements")
    print("4. For detailed function-level analysis: python -m pstats " + stats_file)
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()