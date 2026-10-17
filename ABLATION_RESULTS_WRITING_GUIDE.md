# How to Write Ablation Experiment Results

## Template Structure

### Introduction Paragraph (Problem Setup)
```
We conducted a comprehensive ablation study to validate the design choices 
of our edge caching system. Specifically, we evaluated three key dimensions:
(1) cache placement policies (DDQN vs DQN), (2) request scheduling algorithms 
(EDF vs Round Robin vs FIFO), and (3) cost model formulations (linear vs 
quadratic vs SLA-weighted). The goal was to isolate the contribution of each 
component and demonstrate that our integrated approach achieves superior 
performance compared to individual components.
```

---

## Part 1: Ablation on Cache Placement Policies (DDQN vs DQN)

### Table Format
```markdown
| Metric | DDQN | DQN | Improvement |
|--------|------|-----|-------------|
| Cache Hit Rate (%) | 94.2 | 87.5 | +6.7% |
| Deadline Violation Rate (%) | 3.8 | 9.2 | -5.4pp |
| Total Revenue ($) | 12,450 | 11,200 | +11.2% |
| Forwarding Cost ($) | 2,100 | 3,450 | -39.1% |
| Convergence Episodes | 150 | 180 | -16.7% |
```

### Result Paragraph

**Placement Policy Analysis:**
Our results demonstrate that DDQN significantly outperforms the baseline DQN 
across all metrics. DDQN achieves a 94.2% cache hit rate compared to DQN's 
87.5%, an improvement of 6.7 percentage points. More importantly, the deadline 
violation rate drops from 9.2% to 3.8%, indicating that DDQN's policy 
effectively learns to prioritize critical services for caching. This translates 
to 11.2% higher revenue ($12,450 vs $11,200) while reducing forwarding costs 
by 39.1%. DDQN also converges faster (150 vs 180 episodes), suggesting that 
the double Q-network architecture with target networks provides more stable 
gradient updates. These results validate that the DDQN architecture is necessary 
for the cache placement problem.

---

## Part 2: Ablation on Scheduling Algorithms (EDF vs Round Robin vs FIFO)

### Table Format
```markdown
| Metric | EDF | Round Robin | FIFO | EDF Advantage |
|--------|-----|-------------|------|---------------|
| Deadline Violation Rate (%) | 3.8 | 8.2 | 12.4 | -68.4% |
| Cache Hit Rate (%) | 94.2 | 92.1 | 91.5 | +3.1pp |
| Avg Response Time (ms) | 18.3 | 24.7 | 26.1 | -25.8% |
| Throughput (req/sec) | 1,245 | 1,198 | 1,162 | +7.1% |
| Request Dropped (%) | 2.1 | 5.3 | 7.8 | -72.1% |
| Revenue ($) | 12,450 | 11,890 | 11,340 | +9.8% |
```

### Result Paragraph

**Scheduling Algorithm Analysis:**
EDF significantly outperforms both Round Robin and FIFO scheduling across 
all metrics. The most striking difference is in deadline compliance: EDF 
achieves a violation rate of only 3.8%, compared to 8.2% for Round Robin 
and 12.4% for FIFO—a 68.4% reduction compared to FIFO. This is critical 
in edge caching where SLA violations directly impact revenue. EDF also 
reduces average response time by 25.8% compared to FIFO (18.3ms vs 26.1ms) 
and dropped requests by 72.1% (2.1% vs 7.8%), indicating that prioritizing 
early deadlines prevents queue overflow. While cache hit rates are similar 
across algorithms (94.2% for EDF vs 91.5% for FIFO), the superior deadline 
compliance of EDF translates to 9.8% higher revenue ($12,450 vs $11,340). 
These results confirm that deadline-aware scheduling is essential for 
real-time edge services and that FIFO (a naive baseline) is unsuitable.

---

## Part 3: Ablation on Cost Models

### Table Format
```markdown
| Cost Model | Hit Rate (%) | Violation Rate (%) | Revenue ($) | Final Policy Behavior |
|------------|--------------|-------------------|-------------|----------------------|
| Linear (Current) | 94.2 | 3.8 | 12,450 | Balanced: prioritize high-revenue, respect SLAs |
| No Placement Cost | 96.1 | 5.2 | 11,980 | Overfill cache, thrash memory |
| No Forwarding Cost | 91.4 | 2.1 | 12,100 | Over-forward, waste bandwidth |
| No Violation Penalty | 88.3 | 14.7 | 12,890 | Ignore deadlines, maximize revenue only |
| Quadratic Penalties | 93.7 | 2.9 | 12,210 | Conservative placement, fewer violations |
| SLA-Weighted | 94.5 | 1.2 | 12,380 | Prioritize critical services |
```

### Result Paragraph

**Cost Model Ablation:**
The cost model design critically influences learned cache placement policies. 
Our baseline linear cost model (R - P - F - V) achieves balanced performance: 
94.2% hit rate, 3.8% violations, and $12,450 revenue. Removing placement cost 
(R - F - V) increases hit rate to 96.1% but raises violations to 5.2%, 
indicating the agent learns to cache indiscriminately, consuming memory without 
deadline benefit. Conversely, removing forwarding cost (R - P - V) reduces hit 
rate to 91.4% and revenue to $12,100, as the agent unnecessarily forwards 
requests that could be cached. Removing the violation penalty (R - P - F) 
reveals the importance of deadline awareness: violations spike to 14.7% despite 
achieving the highest nominal revenue ($12,890), demonstrating that ignoring 
SLAs is economically unsustainable. The quadratic penalty model (R - P² - F² - V²) 
reduces violations further to 2.9% at the cost of slightly lower revenue 
($12,210), while the SLA-weighted model (weighted by service criticality) 
achieves the best violation rate (1.2%) at comparable revenue. These results 
validate that all three cost components are necessary and that the linear 
formulation provides the best trade-off between competing objectives.

---

## Part 4: Combined Analysis (Interaction Effects)

### Combined Results Table
```markdown
| Configuration | Hit Rate (%) | Violation (%) | Revenue ($) | Rank |
|---------------|--------------|---------------|-------------|------|
| DDQN + EDF + Linear | 94.2 | 3.8 | 12,450 | **#1** |
| DQN + EDF + Linear | 87.5 | 9.2 | 11,200 | #2 |
| DDQN + Round Robin + Linear | 92.1 | 8.2 | 11,890 | #3 |
| DDQN + FIFO + Linear | 91.5 | 12.4 | 11,340 | #4 |
| DDQN + EDF + No Violation | 88.3 | 14.7 | 12,890 | #5 |
| DDQN + EDF + SLA-Weighted | 94.5 | 1.2 | 12,380 | #6 |
```

### Interaction Analysis Paragraph

**Integrated System Performance:**
To validate that our design choices work synergistically, we evaluated the 
top configurations. The full system (DDQN + EDF + linear cost model) achieves 
the best overall performance: 94.2% hit rate, 3.8% violations, and $12,450 
revenue. This outperforms the isolated best in each dimension (DQN-only yields 
87.5% hit rate; Round Robin scheduling alone gives 8.2% violations; linear 
cost model alone needs complementary policies). Notably, DDQN + FIFO achieves 
only 91.5% hit rate and 12.4% violations, worse than DDQN + EDF, demonstrating 
that no single component can compensate for poor choices in another. The 
SLA-weighted cost model (94.5% hit rate, 1.2% violations) prioritizes deadline 
compliance over revenue, validating the sensitivity of the learned policy to 
cost weights. These results demonstrate positive interaction effects: DDQN 
learns better placement strategies when paired with EDF (deadline-aware 
scheduling) because the state signals it receives include fewer spurious 
deadline violations, making patterns clearer. Conversely, EDF scheduling is 
most effective when services are cached via DDQN's learned policy rather than 
random placement, reducing queue depth and lock contention.

---

## Part 5: Statistical Significance & Confidence Intervals

### Extended Table with Confidence Intervals
```markdown
| Component | Configuration | Hit Rate (%) | 95% CI | Violations (%) | 95% CI | Revenue ($) | 95% CI |
|-----------|----------------|--------------|--------|----------------|--------|-------------|--------|
| Placement | DDQN | 94.2 | ±1.8 | 3.8 | ±0.9 | 12,450 | ±385 |
| (Policy) | DQN | 87.5 | ±2.1 | 9.2 | ±1.4 | 11,200 | ±420 |
| | LRU | 81.3 | ±2.5 | 15.2 | ±2.1 | 10,120 | ±510 |
| Scheduling | EDF | 94.2 | ±1.8 | 3.8 | ±0.9 | 12,450 | ±385 |
| (Algorithm) | Round Robin | 92.1 | ±2.0 | 8.2 | ±1.2 | 11,890 | ±420 |
| | FIFO | 91.5 | ±2.2 | 12.4 | ±1.6 | 11,340 | ±445 |
| Cost Model | Linear (Full) | 94.2 | ±1.8 | 3.8 | ±0.9 | 12,450 | ±385 |
| (Weighting) | No Violation | 88.3 | ±2.3 | 14.7 | ±2.0 | 12,890 | ±410 |
| | SLA-Weighted | 94.5 | ±1.7 | 1.2 | ±0.4 | 12,380 | ±390 |
```

### Statistical Summary Paragraph

**Experimental Robustness:**
All experiments were conducted over 10 independent runs of 100 training episodes 
each, with identical random seeds for request generation but different neural 
network initializations. Reported metrics are means with 95% confidence intervals 
derived from bootstrapped samples. DDQN's superiority over DQN is statistically 
significant (hit rate 94.2% ± 1.8% vs 87.5% ± 2.1%, non-overlapping CI), as is 
EDF's advantage over FIFO (violations 3.8% ± 0.9% vs 12.4% ± 1.6%). The linear 
cost model and SLA-weighted variant show overlapping confidence intervals in 
revenue (12,450 ± 385 vs 12,380 ± 390), suggesting similar economic performance; 
however, the SLA-weighted model significantly reduces violations (1.2% ± 0.4% vs 
3.8% ± 0.9%), making it preferable for deadline-critical applications. The tight 
confidence intervals indicate that our system is stable and reproducible.

---

## Part 6: Comprehensive Summary Paragraph

**Ablation Study Conclusions:**
Our comprehensive ablation study across three key dimensions—cache placement 
(DDQN vs DQN), scheduling (EDF vs Round Robin vs FIFO), and cost models 
(linear vs quadratic vs SLA-weighted)—demonstrates that each component 
contributes meaningfully to system performance and that no single component 
can compensate for poor choices in others. DDQN placement achieves 94.2% 
cache hit rate, 6.7 percentage points higher than DQN, primarily by learning 
to prioritize high-revenue, deadline-critical services. EDF scheduling 
reduces deadline violations by 68.4% compared to FIFO and is essential for 
SLA compliance in real-time edge environments. The linear cost model provides 
optimal balance between revenue maximization and deadline compliance, though 
the SLA-weighted variant may be preferable when service criticality varies. 
Notably, the components exhibit positive synergies: DDQN learns more robust 
policies when paired with EDF scheduling (because state signals are cleaner), 
and EDF is most effective with intelligently-cached services. The full 
integration achieves $12,450 in revenue with only 3.8% deadline violations, 
representing a 9.8% improvement in revenue and 68.4% reduction in violations 
compared to naive baselines. These results strongly support our architectural 
choices and demonstrate the necessity of combining learning-based cache 
placement with deadline-aware scheduling.

---

## Quick Checklist for Your Results

- [ ] What is the baseline? (usually FIFO, LRU, or random)
- [ ] What is the improvement? (absolute and percentage)
- [ ] Are confidence intervals provided? (shows statistical rigor)
- [ ] Did you explain WHY each component matters? (not just what improved)
- [ ] Did you test interactions? (does A+B work better than A or B alone?)
- [ ] Did you measure multiple metrics? (hit rate, violations, revenue, latency, etc.)
- [ ] Are tables sorted by performance? (helps readers quickly identify best config)
- [ ] Did you highlight the key finding? (what surprised you? what's novel?)

---

## Example: How to Structure Your Section

```
## 5. Ablation Study

### 5.1 Cache Placement Policies
[Intro paragraph about why this matters]
[Table: DDQN vs DQN metrics]
[Analysis paragraph: key findings & surprises]

### 5.2 Scheduling Algorithms
[Intro paragraph about EDF vs FIFO vs Round Robin]
[Table: Comparative metrics]
[Analysis paragraph: why EDF wins, implications for edge computing]

### 5.3 Cost Model Ablation
[Intro paragraph about design space]
[Table: Different cost weightings]
[Analysis paragraph: trade-offs between revenue and SLA]

### 5.4 System Integration & Synergies
[Table: Combined results (best of each + full system)]
[Paragraph: interaction effects, why they work together]

### 5.5 Statistical Significance
[Table: Metrics with confidence intervals]
[Paragraph: robustness, reproducibility]

### 5.6 Conclusions
[Summary: which design choices matter most, ranking by impact]
```
