# Ablation Experiment Design: Scheduling & System Components

## Current Implementation
- **Scheduling Algorithm**: EDF (Earliest Deadline First)
- **Cache Placement**: DDQN agent (learns optimal INSERT/DELETE/KEEP actions)
- **Request Distribution**: Zipf (popularity) + Poisson (arrivals)
- **Cost Model**: Revenue - (Placement Cost + Forwarding Cost + Violation Penalty)

---

## Part 1: Alternative Scheduling Algorithms

### Baseline Algorithms (for comparison)

| Algorithm | Mechanism | Pros | Cons |
|-----------|-----------|------|------|
| **FIFO** | Serve requests in arrival order | Simple, fair, baseline | Ignores deadlines; high violations |
| **Priority Queue (by Revenue)** | Serve high-revenue services first | Maximizes immediate revenue | May violate low-revenue SLAs |
| **SJF (Shortest Job First)** | Serve requests with smallest CPU cycles first | Minimizes latency, queue time | Starves long-running services |
| **LDF (Latest Deadline First)** | Opposite of EDF (serve late deadlines first) | Tests EDF necessity | Expected poor performance |
| **Round Robin** | Rotate through services equally | Fair allocation | Ignores both deadline and revenue |

### Advanced Algorithms

| Algorithm | Mechanism | Pros | Cons |
|-----------|-----------|------|------|
| **Weighted Fair Queuing (WFQ)** | Weight by revenue × remaining time | Balances fairness + revenue | Complex scheduling logic |
| **Dynamic Priority** | Priority = revenue / remaining_deadline | Adapts to both SLA and revenue | Higher overhead |
| **Deadline + Laxity** | Laxity = deadline - execution_time | Prioritizes tight schedules | Requires CPU prediction |
| **Age-based Priority** | Older requests get higher priority (anti-starvation) | Prevents indefinite queueing | May violate deadlines |

### Implementation Location
Replace in [edge_caching/control/core/node.py](edge_caching/control/core/node.py#L192-L230):

```python
def _execute_from_queue(self, request, catalogue):
    """Swap out the EDF heappush logic with alternative algorithm"""
    
    # EDF (current)
    heapq.heappush(self._queue, (request.remaining_deadline, request.request_id, request))
    
    # FIFO alternative
    # self._queue.append(request)
    
    # Priority by revenue
    # heapq.heappush(self._queue, (-svc.revenue, request.request_id, request))
    
    # SJF (shortest job first)
    # heapq.heappush(self._queue, (svc.size, request.request_id, request))
```

---

## Part 2: Other Ablation Experiments

### A. Cache Placement Policies

Compare different cache eviction/placement strategies:

| Policy | Mechanism | Expected Behavior |
|--------|-----------|-------------------|
| **DDQN (current)** | Learns optimal placement via DQN | Best performance, adaptive |
| **LRU (Least Recently Used)** | Evict least-recently-accessed service | Baseline, good temporal locality |
| **LFU (Least Frequently Used)** | Evict least-frequently-requested service | Matches Zipf popularity |
| **Greedy (by Revenue)** | Always keep high-revenue services cached | Simple heuristic, static |
| **Random Eviction** | Random placement/eviction | Worst-case baseline |
| **Hybrid (LRU + Revenue)** | Weight by both recency and revenue | Balanced heuristic |

**Experiment Script**: [run_ablation_experiment.py](run_ablation_experiment.py) - modify to swap policies

```python
# Example ablation config
POLICIES = {
    "ddqn": DDQNAgent(...),
    "lru": LRUCachePolicy(...),
    "lfu": LFUCachePolicy(...),
    "greedy": GreedyCachePolicy(...),
    "random": RandomCachePolicy(...),
}
```

---

### B. Reward Function Components

Test impact of each cost component in the DDQN reward:

```
Current: R = Revenue - (Placement Cost + Forwarding Cost + Violation Penalty)
```

| Ablation | Reward Function | Expected Impact |
|----------|-----------------|-----------------|
| **Full Model** | R = Rev - (P + F + V) | Balanced optimization |
| **No Placement Cost** | R = Rev - (F + V) | Overfill caches, higher memory use |
| **No Forwarding Cost** | R = Rev - (P + V) | Forward unnecessarily, waste bandwidth |
| **No Violation Penalty** | R = Rev - (P + F) | Ignore SLAs, low hit rates on deadlines |
| **Revenue Only** | R = Rev | Ignore costs, focus on revenue |
| **No Violations** | R = Rev - (P + F), ignore deadline violations | See baseline deadline behavior |
| **Equal Weights** | R = Rev - α(P + F + V), α varies | Test weight sensitivity |

**Experiment**: Run training with different cost weights:
```python
alpha_placement = 1.0, 0.5, 0.0  # placement cost weight
alpha_forward = 1.0, 0.5, 0.0    # forwarding cost weight
alpha_violation = 1.0, 0.5, 0.0  # violation penalty weight
```

---

### C. Request Pattern Ablation

Test sensitivity to synthetic data parameters:

| Parameter | Values to Test | Expected Impact |
|-----------|----------------|-----------------|
| **Zipf α (popularity skew)** | 0.5, 1.0, 1.2, 1.5, 2.0 | Higher α = more uneven popularity |
| **Poisson λ (arrival rate)** | 1.0, 2.0, 3.0, 5.0 requests/ms | Higher λ = more congestion |
| **Deadline ranges** | Tight (5-40ms), Medium (50-300ms), Loose (200-500ms) | Tests SLA sensitivity |
| **Service count** | 10, 25, 50, 100 services | Tests scalability |
| **Cache capacity** | 5, 10, 20, 50 services/node | Tests capacity constraints |

**Experiment**:
```python
for zipf_alpha in [0.5, 1.0, 1.2, 1.5, 2.0]:
    for poisson_rate in [1.0, 2.0, 3.0]:
        train_ddqn(zipf_alpha, poisson_rate)
        evaluate_performance()
```

---

### D. Network Topology Ablation

Test DDQN generalization across topologies:

| Topology | Config | Expected Behavior |
|----------|--------|-------------------|
| **Current** | 3 levels × 3 children = 27 leaf nodes | Baseline |
| **Shallow** | 2 levels × 10 children = 10 leaf nodes | Fewer hops, different cache strategy |
| **Deep** | 5 levels × 2 children = 32 leaf nodes | More hops, higher forwarding cost |
| **Wide** | 2 levels × 50 children = 50 leaf nodes | Many leaves, wide distribution |
| **Unbalanced** | Mixed children per level | Tests adaptive strategies |

**Experiment**:
```python
topologies = [
    (3, 3),  # current
    (2, 10), # shallow
    (5, 2),  # deep
    (4, 3),  # different
]

for levels, children in topologies:
    train_ddqn(levels=levels, children=children)
    compare_generalization()
```

---

### E. Provisioning Interval Sensitivity

Test how often DDQN should make cache decisions:

| Interval | Requests/Interval | Expected Behavior |
|----------|-------------------|-------------------|
| **Very Fast** | 100 requests | Too frequent, thrashing cache |
| **Fast** | 500 requests | Good responsiveness |
| **Current** | 1000 requests | Balanced |
| **Slow** | 2000 requests | Delayed adaptation, suboptimal |
| **Very Slow** | 5000 requests | Static cache, poor for dynamics |

**Experiment**:
```python
for prov_interval in [100, 500, 1000, 2000, 5000]:
    train_ddqn(provisioning_interval=prov_interval)
    measure_convergence_speed_and_performance()
```

---

### F. Service Characteristics Ablation

Compare uniform vs realistic (Zipf-based) service attributes:

| Variant | Configuration | Expected Impact |
|---------|---------------|-----------------|
| **Realistic (Current)** | Zipf popularity, varied revenue, mixed criticality | Realistic optimization |
| **Uniform Popularity** | All services equally popular | Tests cache fairness |
| **Uniform Revenue** | All services same revenue (focus on SLA) | Revenue-agnostic strategy |
| **Uniform Deadlines** | All services same deadline tiers | Tests without SLA differentiation |
| **Random Attributes** | All randomly assigned | Tests robustness |

---

### G. DDQN Hyperparameter Ablation

Test sensitivity of learning algorithm:

| Hyperparameter | Values | Expected Impact |
|----------------|--------|-----------------|
| **Learning Rate** | 1e-5, 1e-4, 1e-3, 1e-2 | Too low = slow convergence; too high = instability |
| **Epsilon Decay** | 0.995, 0.998, 0.9995 | Fast decay = exploit early; slow = explore longer |
| **Batch Size** | 32, 64, 128, 256 | Larger = stable but slower updates |
| **Replay Buffer Size** | 1k, 5k, 10k, 50k | Larger = better diversity but memory intensive |
| **Target Network Update** | Every 100, 500, 1000 steps | Too frequent = unstable; too slow = stale targets |
| **Discount Factor (γ)** | 0.95, 0.99, 0.999 | Impact on long-term credit assignment |

---

### H. Cost Model Ablation

Test different cost formulations:

| Model | Formula | Expected Behavior |
|-------|---------|-------------------|
| **Current (Linear)** | R - (P + F + V) | Straightforward trade-offs |
| **Non-linear Penalty** | R - (P + F + V²) | Penalize violations more heavily |
| **Quadratic Costs** | R - (P² + F² + V²) | Discourage waste at scale |
| **Time-decay Revenue** | R × e^(-t/T) - (P + F + V) | Older cached content less valuable |
| **SLA-weighted** | R - P - F - (V × SLA_criticality) | Critical services: higher violation cost |

---

## Part 3: Comprehensive Ablation Experiment Design

### Experiment Matrix

```
Dimensions to vary:
1. Scheduling Algorithm (7 options)
2. Cache Policy (6 options)
3. Reward Components (6 options)
4. Zipf α (5 values)
5. Poisson λ (4 values)
6. Topology (5 options)

Total combinations: 7 × 6 × 6 × 5 × 4 × 5 = 25,200 experiments (too many!)

Recommendation: Test independently (factorial ablation):
- Fix A, vary B (holding C, D, E constant)
- Fix B, vary C (holding others constant)
- etc.
```

### Recommended Ablation Suite (Practical)

**Phase 1: Scheduling Algorithms** (1 week)
```python
algorithms = ["EDF", "FIFO", "Priority_Revenue", "SJF", "Round_Robin"]
for algo in algorithms:
    run_100_episodes()
    compare_hit_rate, violation_rate, revenue, cost
```

**Phase 2: Cache Policies** (1 week)
```python
policies = ["DDQN", "LRU", "LFU", "Greedy", "Random"]
for policy in policies:
    run_100_episodes()
    compare_vs_EDF_scheduling
```

**Phase 3: Reward Components** (1 week)
```python
reward_configs = [
    {"P": 1.0, "F": 1.0, "V": 1.0},  # current
    {"P": 0.0, "F": 1.0, "V": 1.0},  # no placement cost
    {"P": 1.0, "F": 0.0, "V": 1.0},  # no forwarding cost
    {"P": 1.0, "F": 1.0, "V": 0.0},  # no violation cost
]
for config in reward_configs:
    run_100_episodes()
    compare_convergence_and_performance
```

**Phase 4: Request Patterns** (1 week)
```python
for zipf_alpha in [0.5, 1.0, 1.2, 1.5, 2.0]:
    for poisson_rate in [1.0, 2.0, 3.0]:
        run_50_episodes()
        plot_sensitivity_heatmap()
```

**Phase 5: Network Topology** (1 week)
```python
topologies = [(3,3), (2,10), (5,2), (4,3), (3,4)]
for topo in topologies:
    train_model_on_topo()
    evaluate_on_other_topos()  # test generalization
```

---

## Expected Outputs

For each ablation experiment, measure:

1. **Performance Metrics**
   - Cache hit rate (%)
   - Deadline violation rate (%)
   - Total revenue ($)
   - Forwarded requests (%)

2. **Efficiency Metrics**
   - Convergence speed (episodes to target performance)
   - Training time (seconds)
   - Memory usage (replay buffer size)

3. **Robustness Metrics**
   - Performance variance across runs
   - Generalization to unseen request patterns
   - Stability of learned policy

4. **Visualizations**
   - Convergence curves for each variant
   - Heatmaps of sensitivity analysis
   - Comparative bar charts
   - Box plots of performance distribution

---

## Implementation Files

- **Current EDF**: [edge_caching/control/core/node.py#L192-L230](edge_caching/control/core/node.py#L192-L230)
- **Cache Policies**: Create `edge_caching/control/policies/` directory with:
  - `lru_policy.py`
  - `lfu_policy.py`
  - `greedy_policy.py`
  - `random_policy.py`
- **Ablation Runner**: [run_ablation_experiment.py](run_ablation_experiment.py) (already exists)
- **Result Analysis**: Create `analyze_ablation_results.py`

---

## Recommended Next Steps

1. **High Priority** (most impact):
   - Scheduling algorithm comparison (EDF vs FIFO vs Priority)
   - Cache policy comparison (DDQN vs LRU vs LFU)
   - Reward component ablation (test cost weights)

2. **Medium Priority** (validation):
   - Request pattern sensitivity (Zipf α, Poisson λ)
   - Network topology generalization

3. **Lower Priority** (optimization):
   - Hyperparameter fine-tuning
   - Cost model variations
