"""
Generate sample_requests CSV for configurable N-ary hierarchical trees.

Usage:
    python generate_requests_3_3.py --levels 4 --children 2 --requests 10000 --output training/data/sample_requests_10000_4_2.csv
    python generate_requests_3_3.py --levels 3 --children 3 --requests 10000 --output training/data/sample_requests_10000_3_3.csv
    python generate_requests_3_3.py --levels 3 --children 4 --requests 10000 --output training/data/sample_requests_10000_3_4.csv

For NUM_LEVELS=L, NUM_CHILDREN=C:
- Total leaf nodes = C^L
- Example: levels=4, children=2 → 2⁴ = 16 leaf nodes (L4_0 through L4_15)
- Example: levels=3, children=3 → 3³ = 27 leaf nodes (L3_0 through L3_26)

Uses Poisson arrival times and Zipf distribution for service popularity.
Ensures ALL leaf nodes receive requests with guaranteed minimum coverage.
"""

import csv
import random
import numpy as np
import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for importing topology
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Default Configuration
DEFAULT_NUM_REQUESTS = 10000
DEFAULT_NUM_SERVICES = 50
DEFAULT_NUM_LEVELS = 4
DEFAULT_NUM_CHILDREN = 2
DEFAULT_POISSON_RATE = 2.0  # requests per ms
DEFAULT_ZIPF_ALPHA = 1.2
DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_TIME_MS = 5000.0  # Total simulation time window
DEFAULT_OUTPUT = "sample_requests_10000_4_2.csv"

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
    random_seed: int = DEFAULT_RANDOM_SEED,
    auto_scale: bool = True
):
    """Generate requests for an N-ary hierarchical tree using actual topology.
    
    Args:
        auto_scale: If True, automatically increase num_requests to maintain 
                   minimum 1000 requests per leaf node for adequate training data.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Import topology to get actual leaf node IDs
    from edge_caching.network.topology import NetworkTopology
    from edge_caching.config import NODE_CACHE_CAP, NODE_CPU_CAPACITY
    
    # Create actual topology to get real node IDs
    print(f"Creating topology to get actual leaf node IDs...")
    topology = NetworkTopology(
        num_levels=num_levels,
        num_children=num_children,
        cache_cap=NODE_CACHE_CAP,
        cpu_capacity=NODE_CPU_CAPACITY
    )
    
    # Get actual leaf node IDs from topology
    leaf_node_ids = [node.node_id for node in topology.leaf_nodes]
    leaf_count = len(leaf_node_ids)
    
    # Check for request sparsity and auto-scale if needed
    RECOMMENDED_MIN_PER_LEAF = 1000  # Minimum for effective caching patterns
    requests_per_leaf = num_requests / leaf_count
    
    print(f"\n{'='*80}")
    print(f"REQUEST SPARSITY ANALYSIS")
    print(f"{'='*80}")
    print(f"Tree configuration: {num_levels} levels × {num_children} children = {leaf_count} leaf nodes")
    print(f"Requested: {num_requests:,} total requests")
    print(f"Distribution: {requests_per_leaf:.1f} requests per leaf node")
    
    if requests_per_leaf < RECOMMENDED_MIN_PER_LEAF:
        recommended_total = leaf_count * RECOMMENDED_MIN_PER_LEAF
        print(f"\n⚠ WARNING: Request sparsity detected!")
        print(f"  Current:     {requests_per_leaf:.0f} requests/node")
        print(f"  Recommended: {RECOMMENDED_MIN_PER_LEAF} requests/node minimum")
        print(f"  Optimal total: {recommended_total:,} requests")
        
        if auto_scale:
            original_num_requests = num_requests
            num_requests = recommended_total
            print(f"\n✓ AUTO-SCALING ENABLED")
            print(f"  Increasing from {original_num_requests:,} to {num_requests:,} requests")
            print(f"  New distribution: {num_requests/leaf_count:.0f} requests/node")
        else:
            print(f"\n⚠ Proceeding with sparse distribution (auto-scale disabled)")
            print(f"  This may result in:")
            print(f"    - Poor cache hit rates (insufficient temporal locality)")
            print(f"    - Undertrained DDQN model (not enough samples per node)")
            print(f"    - Higher drop rates at increased load")
            print(f"\n  To fix: Use --auto-scale or increase --requests to {recommended_total:,}")
    else:
        print(f"✓ Request distribution adequate: {requests_per_leaf:.0f} requests/node")
    
    print(f"{'='*80}\n")
    
    print(f"Generating {num_requests:,} requests for {num_levels},{num_children} tree ({leaf_count} leaf nodes)")
    print(f"Using Poisson (λ={poisson_rate}) and Zipf (α={zipf_alpha})")
    print(f"Leaf nodes: {sorted(leaf_node_ids)}")
    
    # Ensure minimum coverage: each leaf node gets at least some requests
    min_requests_per_leaf = max(10, num_requests // (leaf_count * 4))
    print(f"Guaranteeing minimum {min_requests_per_leaf} requests per leaf node for full coverage...")
    
    # Generate Zipf weights for service popularity
    service_weights = generate_zipf_weights(num_services, zipf_alpha)
    
    # Generate arrival times (Poisson process, normalized to max_time_ms)
    arrival_times = generate_poisson_arrivals(num_requests, poisson_rate, max_time_ms)
    
    requests = []
    request_count = 0
    
    # Phase 1: Guarantee minimum requests for each leaf node
    for leaf_node in leaf_node_ids:
        for _ in range(min_requests_per_leaf):
            request_count += 1
            if request_count > num_requests:
                break
            
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
            
            # Arrival time from Poisson process
            arrival_time_ms = arrival_times[request_count - 1]
            
            requests.append({
                "request_id": request_count,
                "service_id": service_id,
                "deadline_ms": round(deadline_ms, 2),
                "origin_node_id": leaf_node,
                "arrival_time_ms": round(arrival_time_ms, 3),
                "cpu_cycles": cpu_cycles,
            })
        if request_count >= num_requests:
            break
    
    # Phase 2: Fill remaining requests with random distribution
    while request_count < num_requests:
        request_count += 1
        
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
        
        # Select origin node uniformly from all leaf nodes (random distribution)
        origin_node_id = random.choice(leaf_node_ids)
        
        # Arrival time from Poisson process
        arrival_time_ms = arrival_times[request_count - 1]
        
        requests.append({
            "request_id": request_count,
            "service_id": service_id,
            "deadline_ms": round(deadline_ms, 2),
            "origin_node_id": origin_node_id,
            "arrival_time_ms": round(arrival_time_ms, 3),
            "cpu_cycles": cpu_cycles,
        })
    
    # Shuffle to randomize order and avoid sequential node patterns
    random.shuffle(requests)
    
    # Renumber request IDs after shuffling
    for idx, req in enumerate(requests, 1):
        req["request_id"] = idx
    
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
    
    # Coverage validation
    covered_nodes = set(node_counts.keys())
    expected_nodes = set(leaf_node_ids)
    missing_nodes = expected_nodes - covered_nodes
    
    if missing_nodes:
        print(f"  ⚠ WARNING: {len(missing_nodes)} leaf nodes have NO requests:")
        for node in sorted(missing_nodes):
            print(f"    - {node}")
    else:
        print(f"  ✓ SUCCESS: All {leaf_count} leaf nodes have requests!")
    
    # Distribution statistics
    if node_counts:
        counts = list(node_counts.values())
        print(f"  Requests per node - Min: {min(counts)}, Max: {max(counts)}, Avg: {sum(counts)/len(counts):.1f}")
    
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
  # Generate 10K requests for 4-level, 2-children tree (16 leaf nodes) - DEFAULT
  python generate_requests_3_3.py --levels 4 --children 2 --requests 10000 --output training/data/sample_requests_10000_4_2.csv
  
  # Generate 10K requests for 3-level, 3-children tree (27 leaf nodes)
  python generate_requests_3_3.py --levels 3 --children 3 --requests 10000 --output training/data/sample_requests_10000_3_3.csv
  
  # Generate 10K requests for 3-level, 4-children tree (64 leaf nodes)
  python generate_requests_3_3.py --levels 3 --children 4 --requests 10000 --output training/data/sample_requests_10000_3_4.csv
  
  # Use custom Zipf alpha and Poisson rate
  python generate_requests_3_3.py --levels 4 --children 2 --zipf-alpha 1.5 --poisson-rate 3.0
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
    parser.add_argument("--auto-scale", action="store_true", default=True,
                        help="Automatically scale request count to maintain 1000+ requests per leaf (default: True)")
    parser.add_argument("--no-auto-scale", action="store_false", dest="auto_scale",
                        help="Disable auto-scaling (use exact --requests value)")
    
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
        random_seed=args.seed,
        auto_scale=args.auto_scale
    )


if __name__ == "__main__":
    main()