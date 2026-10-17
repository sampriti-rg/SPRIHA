# -*- coding: utf-8 -*-
"""Verify all four fixes are correctly in place."""
import sys
sys.path.insert(0, ".")

from edge_caching.config import MAX_FORWARD_NORM, MAX_EXECUTE_NORM, MEMORY_SIZE, PROVISIONING_INTERVAL
from edge_caching.agents.ddqn_agent import DDQNAgent
from edge_caching.core.service import Service

svc = Service(service_id=0, size=100.0, popularity=0.5,
              criticality="standard", revenue=5.0)

stats = {
    "forwarded_count"  : {0: PROVISIONING_INTERVAL},
    "executed_count"   : {0: PROVISIONING_INTERVAL},
    "cache_utilization": 0.5,
    "free_capacity"    : 5000.0,
    "cached_services"  : set(),
}

vec = DDQNAgent.build_state_vector(svc, stats)

print("=" * 60)
print("Fix 1 - State feature saturation")
print(f"  MAX_FORWARD_NORM = {MAX_FORWARD_NORM}  (must equal {PROVISIONING_INTERVAL})"
      f"  {'PASS' if MAX_FORWARD_NORM == PROVISIONING_INTERVAL else 'FAIL'}")
print(f"  MAX_EXECUTE_NORM = {MAX_EXECUTE_NORM}  (must equal {PROVISIONING_INTERVAL})"
      f"  {'PASS' if MAX_EXECUTE_NORM == PROVISIONING_INTERVAL else 'FAIL'}")
print(f"  feature[4] forwarded_rate = {vec[4]:.4f}  (expected 1.0)  "
      f"{'PASS' if abs(vec[4] - 1.0) < 1e-6 else 'FAIL'}")
print(f"  feature[5] executed_rate  = {vec[5]:.4f}  (expected 1.0)  "
      f"{'PASS' if abs(vec[5] - 1.0) < 1e-6 else 'FAIL'}")

print()
print("Fix 2 - Reward scale (r_max = 30)")
reward_pos = DDQNAgent.compute_reward(svc, delta_served=1, delta_executed=1,
                                      delta_forwarded=0, placement_cost=0.0)
reward_neg = DDQNAgent.compute_reward(svc, delta_served=0, delta_executed=0,
                                      delta_forwarded=5, placement_cost=0.0)
print(f"  reward (1 served, 1 exec)   = {reward_pos:.4f}  (expected > 0.05)"
      f"  {'PASS' if reward_pos > 0.05 else 'FAIL'}")
print(f"  reward (5 forwarded)        = {reward_neg:.4f}  (expected < 0.0)"
      f"  {'PASS' if reward_neg < 0.0 else 'FAIL'}")
print(f"  reward clipped to [-1, +1]  "
      f"  {'PASS' if -1.0 <= reward_pos <= 1.0 and -1.0 <= reward_neg <= 1.0 else 'FAIL'}")

print()
print("Fix 3 - Replay buffer capacity")
transitions_per_ep = 14 * 50 * (10000 // PROVISIONING_INTERVAL)
print(f"  MEMORY_SIZE           = {MEMORY_SIZE}")
print(f"  transitions/episode   = {transitions_per_ep}")
print(f"  buffer holds episodes = {MEMORY_SIZE / transitions_per_ep:.2f}")
print(f"  MEMORY_SIZE >= transitions/ep"
      f"  {'PASS' if MEMORY_SIZE >= transitions_per_ep else 'FAIL'}")

print()
print("Fix 4 - Hit rate metric (verified in train.py source)")
import ast, os
src = open(os.path.join("training", "train.py")).read()
leaf_counter_present = "ep_leaf_executed" in src
leaf_passed_to_end   = "total_executed   = ep_leaf_executed" in src
leaf_rate_printed    = "leaf_hit_rate" in src
print(f"  ep_leaf_executed counter present  {'PASS' if leaf_counter_present else 'FAIL'}")
print(f"  leaf count passed to end_episode  {'PASS' if leaf_passed_to_end else 'FAIL'}")
print(f"  leaf_hit_rate printed in summary  {'PASS' if leaf_rate_printed else 'FAIL'}")

print()
all_pass = all([
    MAX_FORWARD_NORM == PROVISIONING_INTERVAL,
    MAX_EXECUTE_NORM == PROVISIONING_INTERVAL,
    abs(vec[4] - 1.0) < 1e-6,
    abs(vec[5] - 1.0) < 1e-6,
    reward_pos > 0.05,
    reward_neg < 0.0,
    MEMORY_SIZE >= transitions_per_ep,
    leaf_counter_present,
    leaf_passed_to_end,
    leaf_rate_printed,
])
print("=" * 60)
print(f"  OVERALL: {'ALL FIXES VERIFIED - safe to retrain' if all_pass else 'ONE OR MORE FIXES MISSING'}")
print("=" * 60)
