import sys
sys.path.insert(0, ".")

from edge_caching.config import (
    NODE_CACHE_CAP, WARMUP_STEPS, EPSILON_DECAY,
    REQUESTS_PER_EPISODE, PROVISIONING_INTERVAL,
    MEMORY_SIZE, LAMBDA_PLACEMENT
)
print("NODE_CACHE_CAP       :", NODE_CACHE_CAP,        " (expect 10)")
print("WARMUP_STEPS         :", WARMUP_STEPS,          " (expect 1000)")
print("EPSILON_DECAY        :", EPSILON_DECAY,         " (expect 0.994)")
print("REQUESTS_PER_EPISODE :", REQUESTS_PER_EPISODE,  " (expect 500)")
print("PROVISIONING_INTERVAL:", PROVISIONING_INTERVAL, " (expect 50)")
print("MEMORY_SIZE          :", MEMORY_SIZE,           " (expect 20000)")
print("LAMBDA_PLACEMENT     :", LAMBDA_PLACEMENT,      " (expect 0.1)")

from edge_caching.utils.simulation.request_generator import _DEADLINE_RANGES
print()
print("_DEADLINE_RANGES:", _DEADLINE_RANGES)
