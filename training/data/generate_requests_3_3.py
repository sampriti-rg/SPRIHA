"""
Generate sample_requests_10000_3_3.csv for a 3-level, 3-children hierarchical tree.

For NUM_LEVELS=3, NUM_CHILDREN=3:
- Level 1: 3 nodes (L1_0, L1_1, L1_2)
- Level 2: 9 nodes (L2_0 through L2_8)
- Level 3 (leaf): 27 nodes (L3_0 through L3_26)

Uses Poisson arrival times and Zipf distribution for service popularity.
"""

import csv
import random
import numpy as np
from typing import List, Tuple

# Configuration
NUM_REQUESTS = 20000
NUM_SERVICES = 50
NUM_LEVELS = 3
NUM_CHILDREN = 3
POISSON_ARRIVAL_RATE = 2.0  # requests per ms
ZIPF_ALPHA = 1.2
RANDOM_SEED = 42
MAX_TIME_MS = 5000.0  # Total simulation time window

# Calculate leaf nodes for 3,3 tree
# Level 1: 3 nodes, Level 2: 9 nodes, Level 3: 27 nodes (leaves)
LEAF_COUNT = NUM_CHILDREN ** NUM_LEVELS  # 3^3 = 27
LEAF_NODE_IDS = [f"L3_{i}" for i in range(LEAF_COUNT)]

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


def generate_requests_3_3(output_path: str):
    """Generate exactly 10000 requests for a 3,3 hierarchical tree."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    print(f"Generating {NUM_REQUESTS} requests for 3,3 tree ({LEAF_COUNT} leaf nodes)")
    print(f"Using Poisson (λ={POISSON_ARRIVAL_RATE}) and Zipf (α={ZIPF_ALPHA})")
    
    # Generate Zipf weights for service popularity
    service_weights = generate_zipf_weights(NUM_SERVICES, ZIPF_ALPHA)
    
    # Generate arrival times (Poisson process, normalized to MAX_TIME_MS)
    arrival_times = generate_poisson_arrivals(NUM_REQUESTS, POISSON_ARRIVAL_RATE, MAX_TIME_MS)
    
    requests = []
    
    for req_id in range(1, NUM_REQUESTS + 1):
        # Select service using Zipf distribution
        service_id = np.random.choice(NUM_SERVICES, p=service_weights)
        
        # Get service criticality
        criticality = SERVICE_CRITICALITY[service_id]
        
        # Generate deadline based on criticality
        deadline_min, deadline_max = DEADLINE_RANGES[criticality]
        deadline_ms = random.uniform(deadline_min, deadline_max)
        
        # Generate CPU cycles based on criticality
        cpu_min, cpu_max = SERVICE_CPU_CYCLES[criticality]
        cpu_cycles = random.randint(cpu_min, cpu_max)
        
        # Select origin node uniformly from all leaf nodes
        origin_node_id = random.choice(LEAF_NODE_IDS)
        
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
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(requests)
    
    print(f"✓ Generated {len(requests)} requests")
    print(f"✓ Saved to {output_path}")
    
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
    print(f"  Unique leaf nodes: {len(node_counts)} (expected: {LEAF_COUNT})")
    print(f"  Top 5 most popular services: {sorted(service_counts.items(), key=lambda x: x[1], reverse=True)[:5]}")
    print(f"  Avg requests per node: {NUM_REQUESTS / LEAF_COUNT:.1f}")
    print(f"  Time window: 0 - {MAX_TIME_MS} ms")
    print(f"  Avg arrival rate: {NUM_REQUESTS / MAX_TIME_MS:.2f} req/ms")


if __name__ == "__main__":
    output_path = "sample_requests_10000_3_3.csv"
    generate_requests_3_3(output_path)