"""
Generate sample_requests CSV using actual topology node IDs.

This ensures 100% compatibility between generated requests and the topology.
"""

import csv
import random
import numpy as np
import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path to import edge_caching modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edge_caching.network.topology import NetworkTopology
from edge_caching.config import NODE_CACHE_CAP, NODE_CPU_CAPACITY

# Default Configuration
DEFAULT_NUM_REQUESTS = 10000
DEFAULT_NUM_SERVICES = 50
DEFAULT_NUM_LEVELS = 3
DEFAULT_NUM_CHILDREN = 2
DEFAULT_POISSON_RATE = 2.0
DEFAULT_ZIPF_ALPHA = 1.2
DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_TIME_MS = 5000.0

# Criticality-aware deadline ranges [ms]
DEADLINE_RANGES = {
    "mission_critical": (5.0, 40.0),
    "high": (50.0, 150.0),
    "standard": (100.0, 300.0),
    "best_effort": (200.0, 500.0),
}

# Service characteristics
SERVICE_CRITICALITY = [
    "mission_critical", "mission_critical", "mission_critical",
    "high", "high", "high", "high", "high",
    "standard", "standard", "standard", "standard", "standard",
    "standard", "standard", "standard", "standard", "standard",
    "standard", "standard", "standard", "standard", "standard",
    "best_effort", "best_effort", "best_effort", "best_effort",
    "best_effort", "best_effort", "best_effort", "best_effort",
    "best_effort", "best_effort", "best_effort", "best_effort",
    "best_effort", "best_effort", "best_effort", "best_effort",
    "best_effort", "best_effort", "best_effort", "best_effort",
    "best_effort", "best_effort", "best_effort", "best_effort",
    "best_effort", "best_effort", "best_effort"
]

# CPU cycles per service
SERVICE_CPU_CYCLES = {
    "mission_critical": (50000, 200000),
    "high": (10000, 50000),
    "standard": (5000, 30000),
    "best_effort": (500, 5000),
}


def generate_zipf_weights(num_services: int, alpha: float) -> np.ndarray:
    """Generate Zipf probability weights."""
    ranks = np.arange(1, num_services + 1)
    weights = 1.0 / (ranks ** alpha)
    weights /= weights.sum()
    return weights


def generate_poisson_arrivals(n: int, rate: float, max_time: float) -> np.ndarray:
    """Generate exactly n arrival times with Poisson spacing."""
    inter_arrivals = np.random.exponential(scale=1.0 / rate, size=n)
    cumulative_times = np.cumsum(inter_arrivals)
    arrival_times = (cumulative_times / cumulative_times[-1]) * max_time
    return arrival_times


def generate_requests(
    num_requests: int,
    num_services: int,
    num_levels: int,
    num_children: int,
    output_path: str,
    poisson_rate: float = DEFAULT_POISSON_RATE,
    zipf_alpha: float = DEFAULT_ZIPF_ALPHA,
    max_time_ms: float = DEFAULT_MAX_TIME_MS,
    random_seed: int = DEFAULT_RANDOM_SEED
):
    """Generate requests for an N-ary hierarchical tree using actual topology."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create actual topology to get real leaf node IDs
    topology = NetworkTopology(
        num_levels=num_levels,
        num_children=num_children,
        cache_cap=NODE_CACHE_CAP,
        cpu_capacity=NODE_CPU_CAPACITY
    )
    
    # Get actual leaf node IDs from topology
    leaf_node_ids = [node.node_id for node in topology.leaf_nodes]
    leaf_count = len(leaf_node_ids)
    
    print(f"Generating {num_requests} requests for {num_levels},{num_children} tree")
    print(f"Topology has {leaf_count} leaf nodes: {sorted(leaf_node_ids)}")
    print(f"Using Poisson (λ={poisson_rate}) and Zipf (α={zipf_alpha})")
    
    # Generate Zipf weights for service popularity
    service_weights = generate_zipf_weights(num_services, zipf_alpha)
    
    # Generate arrival times
    arrival_times = generate_poisson_arrivals(num_requests, poisson_rate, max_time_ms)
    
    requests = []
    
    for req_id in range(1, num_requests + 1):
        # Select service using Zipf distribution
        service_id = np.random.choice(num_services, p=service_weights)
        
        # Get service criticality
        criticality = SERVICE_CRITICALITY[service_id] if service_id < len(SERVICE_CRITICALITY) else "best_effort"
        
        # Generate deadline based on criticality
        deadline_min, deadline_max = DEADLINE_RANGES[criticality]
        deadline_ms = random.uniform(deadline_min, deadline_max)
        
        # Generate CPU cycles based on criticality
        cpu_min, cpu_max = SERVICE_CPU_CYCLES[criticality]
        cpu_cycles = random.randint(cpu_min, cpu_max)
        
        # Select origin node uniformly from all leaf nodes
        origin_node_id = random.choice(leaf_node_ids)
        
        # Arrival time from Poisson process
        arrival_time_ms = arrival_times[req_id - 1]
        
        requests.append({
            "request_id": req_id,
            "service_id": service_id,
            "deadline_ms": round(deadline_ms, 2),
            "origin_node_id": origin_node_id,
            "arrival_time_ms": round(arrival_time_ms, 3),
            "cpu_cycles": cpu_cycles,
        })
    
    # Write to CSV
    fieldnames = ["request_id", "service_id", "deadline_ms", "origin_node_id", "arrival_time_ms", "cpu_cycles"]
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(requests)
    
    print(f"\n✓ Generated {len(requests)} requests")
    print(f"✓ Saved to {output_file}")
    
    # Print statistics
    service_counts = {}
    node_counts = {}
    
    for req in requests:
        sid = req["service_id"]
        nid = req["origin_node_id"]
        service_counts[sid] = service_counts.get(sid, 0) + 1
        node_counts[nid] = node_counts.get(nid, 0) + 1
    
    print(f"\nStatistics:")
    print(f"  Unique services: {len(service_counts)}")
    print(f"  Unique leaf nodes: {len(node_counts)} (expected: {leaf_count})")
    print(f"  All leaf nodes covered: {'✓' if len(node_counts) == leaf_count else '✗'}")
    print(f"  Top 5 most popular services: {sorted(service_counts.items(), key=lambda x: x[1], reverse=True)[:5]}")
    print(f"  Requests per node (min/avg/max): {min(node_counts.values())}/{num_requests/leaf_count:.1f}/{max(node_counts.values())}")
    print(f"  Time window: 0 - {max_time_ms} ms")
    print(f"  Avg arrival rate: {num_requests / max_time_ms:.2f} req/ms")


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample requests CSV using actual topology node IDs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 10K requests for 3-level, 2-children tree
  python generate_requests_with_topology.py --levels 3 --children 2 --requests 10000 --output training/data/sample_requests_10000_3_2.csv
  
  # Generate 10K requests for 3-level, 3-children tree
  python generate_requests_with_topology.py --levels 3 --children 3 --requests 10000 --output training/data/sample_requests_10000_3_3.csv
        """
    )
    
    parser.add_argument("--levels", "-l", type=int, default=DEFAULT_NUM_LEVELS,
                        help=f"Number of tree levels (default: {DEFAULT_NUM_LEVELS})")
    parser.add_argument("--children", "-c", type=int, default=DEFAULT_NUM_CHILDREN,
                        help=f"Number of children per node (default: {DEFAULT_NUM_CHILDREN})")
    parser.add_argument("--requests", "-r", type=int, default=DEFAULT_NUM_REQUESTS,
                        help=f"Number of requests to generate (default: {DEFAULT_NUM_REQUESTS})")
    parser.add_argument("--services", "-s", type=int, default=DEFAULT_NUM_SERVICES,
                        help=f"Number of services (default: {DEFAULT_NUM_SERVICES})")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output CSV filename")
    parser.add_argument("--poisson-rate", type=float, default=DEFAULT_POISSON_RATE,
                        help=f"Poisson arrival rate in req/ms (default: {DEFAULT_POISSON_RATE})")
    parser.add_argument("--zipf-alpha", type=float, default=DEFAULT_ZIPF_ALPHA,
                        help=f"Zipf distribution alpha parameter (default: {DEFAULT_ZIPF_ALPHA})")
    parser.add_argument("--max-time", type=float, default=DEFAULT_MAX_TIME_MS,
                        help=f"Maximum simulation time in ms (default: {DEFAULT_MAX_TIME_MS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED,
                        help=f"Random seed for reproducibility (default: {DEFAULT_RANDOM_SEED})")
    
    args = parser.parse_args()
    
    if args.levels < 1:
        parser.error("--levels must be at least 1")
    if args.children < 2:
        parser.error("--children must be at least 2")
    if args.requests < 1:
        parser.error("--requests must be at least 1")
    if args.services < 1:
        parser.error("--services must be at least 1")
    
    generate_requests(
        num_requests=args.requests,
        num_services=args.services,
        num_levels=args.levels,
        num_children=args.children,
        output_path=args.output,
        poisson_rate=args.poisson_rate,
        zipf_alpha=args.zipf_alpha,
        max_time_ms=args.max_time,
        random_seed=args.seed
    )


if __name__ == "__main__":
    main()
