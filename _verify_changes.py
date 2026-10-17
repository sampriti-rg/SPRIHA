import inspect
from edge_caching.agents.ddqn_agent import DDQNAgent
from edge_caching.core.node import EdgeNode
from edge_caching.control.orchestrator import EdgeOrchestrator
from edge_caching.core.service import Service

sig = inspect.signature(DDQNAgent.compute_reward)
print("compute_reward params:", list(sig.parameters.keys()))

n = EdgeNode("L1_0", 1)
print("served_count attr:", hasattr(n, "served_count"))
snap = n.get_stats_snapshot()
print("snapshot keys:", list(snap.keys()))

svc = Service(0, 2, 0.7, "high", 5.0)

r_cached_congested = DDQNAgent.compute_reward(
svc, delta_served=5, delta_executed=0, delta_forwarded=0,
placement_cost=0.0)
print("cached+congested (served=5, exe=0, fwd=0):", round(r_cached_congested, 4), "-> should be POSITIVE (placement correct)")

r_not_cached = DDQNAgent.compute_reward(
svc, delta_served=0, delta_executed=0, delta_forwarded=5,
placement_cost=0.0)
print("not cached       (served=0, exe=0, fwd=5):", round(r_not_cached, 4), "-> should be NEGATIVE (INSERT needed)")

r_cached_healthy = DDQNAgent.compute_reward(
svc, delta_served=5, delta_executed=5, delta_forwarded=0,
placement_cost=0.0)
print("cached+healthy   (served=5, exe=5, fwd=0):", round(r_cached_healthy, 4), "-> should be MOST POSITIVE")

print("\nAll checks passed." if (r_cached_congested > 0 and r_not_cached < 0 and r_cached_healthy > r_cached_congested) else "\nSOMETHING WRONG")
