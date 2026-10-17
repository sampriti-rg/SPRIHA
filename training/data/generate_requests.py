"""
Generate sample_requests CSV for configurable N-ary hierarchical trees.

Usage:
    python generate_requests.py --levels 3 --children 4 --requests 10000 --output sample_requests_3_4.csv
    python generate_requests.py --levels 3 --children 3 --requests 5000 --output sample_requests_3_3_5k.csv
    python generate_requests.py --levels 2 --children 5 --requests 20000 --output sample_requests_2_5_20k.csv

For NUM_LEVELS=L, NUM_CHILDREN=C:
- Total leaf nodes = C^L
- Example: levels=3, children=4 → 4³ = 64 leaf nodes (L3_0 through L3_63)

Uses Poisson arrival times and Zipf distribution for service popularity.
"""

import csv
import random
import numpy as np
import argparse
from pathlib import Path

# Default Configuration
DEFAULT_NUM_REQUESTS = 10000
DEFAULT_NUM_SERVICES = 50
DEFAULT_NUM_LEVELS = 3
DEFAULT_NUM_CHILDREN = 4
DEFAULT_POISSON_RATE = 2.0  # requests per ms
DEFAULT_ZIPF_ALPHA = 1.2
DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_TIME_MS = 5000.0  # Total simulation time window
DEFAULT_OUTPUT = "sample_requests_10000_3_4.csv"

# Criticality-aware deadline ranges [ms]
DEADLINE_RANGES = {
    "mission_critical": (5.0, 40.0),
    "high": (50.0, 150.0),
    "standard": (100.0, 300.0),
    "best_effort": (200.0, 500.0),
}

# Service characteristics (matching typical edge caching scenarios)
SERVICE_CRITICALITY = [
    "mission_critical", "mission_critical", "mission_critical",  # 0-2: VoIP, VideoStreaming, etc.
    "high", "high", "high", "high", "high",  # 3-7: Interactive services
    "standard", "standard", "standard", "standard", "standard",  # 8-12: Regular content
    "standard", "standard", "standard", "standard", "standard",  # 13-17
    "standard", "standard", "standard", "standard", "standard",  # 18-22
    "best_effort", "best_effort", "best_effort", "best_effort",  # 23-26: Background tasks
    "best_effort", "best_effort", "best_effort", "best_effort",  # 27-30
    "best_effort", "best_effort", "best_effort", "best_effort",  # 31-34
    "best_effort", "best_effort", "best_effort", "best_effort",  # 35-38
    "best_effort", "best_effort", "best_effort", "best_effort",  # 39-42
    "best_effort", "best_effort", "best_effort", "best_effort",  # 43-46
    "best_effort", "best_effort", "best_effort"  # 47-49
]

# CPU cycles per service (mission_critical = higher, best_effort = lower)
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
    """
    Generate exactly n arrival times with Poisson spacing.
    
    Uses exponential inter-arrival times, then normalizes to fit within [0, max_time).
    This guarantees exactly n requests while maintaining Poisson-like spacing.
    """
    # Generate exponential inter-arrival times
    inter_arrivals = np.random.exponential(scale=1.0 / rate, size=n)
    
    # Cumulative sum gives arrival times
    cumulative_times = np.cumsum(inter_arrivals)
    
    # Normalize to fit within [0, max_time)
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
    """Generate requests for an N-ary hierarchical tree."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Calculate leaf nodes
    leaf_count = num_children ** num_levels
    leaf_node_ids = [f"L{num_levels}_{i}" for i in range(leaf_count)]
    
    print(f"Generating {num_requests} requests for {num_levels},{num_children} tree ({leaf_count} leaf nodes)")
    print(f"Using Poisson (λ={poisson_rate}) and Zipf (α={zipf_alpha})")
    
    # Generate Zipf weights for service popularity
    service_weights = generate_zipf_weights(num_services, zipf_alpha)
    
    # Generate arrival times (Poisson process, normalized to max_time_ms)
    arrival_times = generate_poisson_arrivals(num_requests, poisson_rate, max_time_ms)
    
    requests = []
    
    for req_id in range(1, num_requests + 1):
        # Select service using Zipf distribution
        service_id = np.random.choice(num_services, p=service_weights)
        
        # Get service criticality (handle services beyond template)
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
    
    print(f"✓ Generated {len(requests)} requests")
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
    print(f"  Top 5 most popular services: {sorted(service_counts.items(), key=lambda x: x[1], reverse=True)[:5]}")
    print(f"  Avg requests per node: {num_requests / leaf_count:.1f}")
    print(f"  Time window: 0 - {max_time_ms} ms")
    print(f"  Avg arrival rate: {num_requests / max_time_ms:.2f} req/ms")


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample requests CSV for N-ary hierarchical edge caching trees",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 10K requests for 3-level, 4-children tree (64 leaf nodes)
  python generate_requests.py --levels 3 --children 4 --requests 10000 --output sample_requests_3_4.csv
  
  # Generate 5K requests for 3-level, 3-children tree (27 leaf nodes)
  python generate_requests.py --levels 3 --children 3 --requests 5000 --output sample_requests_3_3_5k.csv
  
  # Generate 20K requests for 2-level, 5-children tree (25 leaf nodes)
  python generate_requests.py --levels 2 --children 5 --requests 20000 --output sample_requests_2_5_20k.csv
  
  # Use custom Zipf alpha and Poisson rate
  python generate_requests.py --levels 3 --children 4 --zipf-alpha 1.5 --poisson-rate 3.0
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
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output CSV filename (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--poisson-rate", type=float, default=DEFAULT_POISSON_RATE,
                        help=f"Poisson arrival rate in req/ms (default: {DEFAULT_POISSON_RATE})")
    parser.add_argument("--zipf-alpha", type=float, default=DEFAULT_ZIPF_ALPHA,
                        help=f"Zipf distribution alpha parameter (default: {DEFAULT_ZIPF_ALPHA})")
    parser.add_argument("--max-time", type=float, default=DEFAULT_MAX_TIME_MS,
                        help=f"Maximum simulation time in ms (default: {DEFAULT_MAX_TIME_MS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED,
                        help=f"Random seed for reproducibility (default: {DEFAULT_RANDOM_SEED})")
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.levels < 1:
        parser.error("--levels must be at least 1")
    if args.children < 2:
        parser.error("--children must be at least 2")
    if args.requests < 1:
        parser.error("--requests must be at least 1")
    if args.services < 1:
        parser.error("--services must be at least 1")
    
    leaf_count = args.children ** args.levels
    print(f"Tree configuration: {args.levels} levels, {args.children} children → {leaf_count} leaf nodes\n")
    
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