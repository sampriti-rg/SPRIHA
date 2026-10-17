# Evaluating EDF vs Round Robin on 3-2 Tree with Sample Data

## Current Status

**Config detected:**
- Current SCHEDULING_ALGORITHM: `ROUND_ROBIN`
- Tree: 3 levels × 2 children (3-2)
- Services: sample_services.csv ✓ Available
- Requests: sample_requests_10000_3_3.csv ✓ Available (for 3-3 tree)

---

## Best Script for Your Ablation: `inference.py`

**Why inference.py is best:**
- ✓ Loads finetuned checkpoints easily
- ✓ Doesn't re-train (just evaluates)
- ✓ Supports custom requests per episode
- ✓ Flexible episode count
- ✓ Produces clean metrics
- ✓ Saves CSV reports

`finetunehighload.py` is for high-load training refinement, not evaluation.

---

## Step-by-Step: Run Ablation Experiment

### **Step 1: Check Available Data**

```bash
ls training/data/sample_requests_10000_3_*.csv
```

**Expected output:**
```
sample_requests_10000_3_3.csv          (✓ 3 levels × 3 children)
sample_requests_10000_3_3_complete.csv 
sample_requests_10000_3_3_fixed.csv
sample_requests_10000_3_4.csv          (✓ 3 levels × 4 children)
```

**Problem:** No 3-2 requests file found!

You have:
- sample_services.csv ✓
- sample_requests for 3-3, 3-4, 3-5 trees
- **BUT NOT for 3-2 tree**

---

### **Step 2: Generate 10000 Requests for 3-2 Tree (Required)**

```bash
python generate_requests_3_3.py \
    --levels 3 \
    --children 2 \
    --requests 10000 \
    --output training/data/sample_requests_10000_3_2.csv
```

This creates requests matching your topology: 3 levels, 2 children per node = 8 leaf nodes.

---

### **Step 3: Run EDF Baseline (Control)**

**Modify config.py:**
```python
SCHEDULING_ALGORITHM = "EDF"
```

**Run inference:**
```bash
python training/inference.py \
    --checkpoint checkpoints/ddqn_finetuned_resumed_6_2_400ep_best.pt \
    --episodes 5 \
    --requests 10000 \
    --seed 42
```

This will:
- Load the finetuned model
- Create 3-2 topology (from config)
- Use EDF scheduling (from config)
- Process 10000 requests over 5 episodes (2000/ep)
- Save results to `results/inference_requests.csv`
- Output metrics to console

**Expected output:**
```
Ep        Requests   Executed   HitRate   Dropped  Avg Latency  Leaf%   Cloud%
1          10000       9500     95.0%       500     18.23ms    70.5%    3.2%
2          10000       9520     95.2%       480     17.85ms    71.2%    3.0%
...
```

**Save results:**
```bash
cp results/inference_requests.csv results/ablation_edf_10000_3_2.csv
cp results/inference_report.txt results/ablation_edf_10000_3_2_report.txt
```

---

### **Step 4: Run Round Robin Variant (Treatment)**

**Modify config.py:**
```python
SCHEDULING_ALGORITHM = "ROUND_ROBIN"
```

**Clear cache:**
```bash
Remove-Item -Path edge_caching -Recurse -Include __pycache__ -Force
```

**Run inference (exact same command):**
```bash
python training/inference.py \
    --checkpoint checkpoints/ddqn_finetuned_resumed_6_2_400ep_best.pt \
    --episodes 5 \
    --requests 10000 \
    --seed 42
```

**Save results:**
```bash
cp results/inference_requests.csv results/ablation_roundrobin_10000_3_2.csv
cp results/inference_report.txt results/ablation_roundrobin_10000_3_2_report.txt
```

---

### **Step 5: Compare Results**

**Metrics to compare:**

| Metric | File Location | Command |
|--------|-------------|---------|
| Hit Rate | inference_requests.csv | `grep "executed" ablation_*_10000_3_2.csv \| wc -l` |
| Violations | inference_requests.csv | `grep "dropped" ablation_*_10000_3_2.csv \| wc -l` |
| Avg Latency | inference_requests.csv | Column `latency_consumed_ms` |
| Cache Utilization | inference_report.txt | Section [3] FINAL CACHE STATE |

**Python comparison script:**

```python
import pandas as pd

edf_df = pd.read_csv("results/ablation_edf_10000_3_2.csv")
rr_df = pd.read_csv("results/ablation_roundrobin_10000_3_2.csv")

print("="*70)
print("EDF vs ROUND ROBIN: 10000 Requests on 3-2 Tree")
print("="*70)

for df, name in [(edf_df, "EDF"), (rr_df, "ROUND_ROBIN")]:
    executed = len(df[df["status"] == "executed"])
    dropped = len(df[df["status"] == "dropped"])
    hit_rate = executed / len(df) * 100
    avg_latency = df[df["status"] == "executed"]["latency_consumed_ms"].astype(float).mean()
    
    print(f"\n{name}:")
    print(f"  Hit Rate:      {hit_rate:.2f}%")
    print(f"  Violations:    {dropped} ({dropped/len(df)*100:.2f}%)")
    print(f"  Avg Latency:   {avg_latency:.2f} ms")
    print(f"  Total:         {len(df)}")
```

---

## Complete Workflow Script

```bash
#!/bin/bash
# Ablation: EDF vs Round Robin on 3-2 tree with 10000 requests

CHECKPOINT="checkpoints/ddqn_finetuned_resumed_6_2_400ep_best.pt"
EPISODES=5
REQUESTS=10000
SEED=42

echo "Step 1: Generate 10000 requests for 3-2 tree"
python generate_requests_3_3.py \
    --levels 3 --children 2 --requests $REQUESTS \
    --output training/data/sample_requests_10000_3_2.csv

echo -e "\nStep 2: Run EDF baseline"
# Modify config: SCHEDULING_ALGORITHM = "EDF"
python training/inference.py \
    --checkpoint $CHECKPOINT \
    --episodes $EPISODES \
    --requests $REQUESTS \
    --seed $SEED
cp results/inference_requests.csv results/ablation_edf_10000_3_2.csv

echo -e "\nStep 3: Clear Python cache"
Remove-Item -Path edge_caching -Recurse -Include __pycache__ -Force

echo -e "\nStep 4: Run Round Robin variant"
# Modify config: SCHEDULING_ALGORITHM = "ROUND_ROBIN"
python training/inference.py \
    --checkpoint $CHECKPOINT \
    --episodes $EPISODES \
    --requests $REQUESTS \
    --seed $SEED
cp results/inference_requests.csv results/ablation_roundrobin_10000_3_2.csv

echo -e "\nStep 5: Results saved"
echo "  EDF:        results/ablation_edf_10000_3_2.csv"
echo "  Round Robin: results/ablation_roundrobin_10000_3_2.csv"
```

---

## Expected Differences

| Metric | EDF | Round Robin | Impact |
|--------|-----|-------------|--------|
| **Hit Rate** | ~95% | ~93% | EDF prioritizes deadline-critical services |
| **Deadline Violations** | 3-5% | 10-15% | EDF minimizes deadline misses |
| **Avg Latency** | Lower | Higher | EDF serves urgent requests first |
| **Leaf % Execution** | 70-75% | 65-70% | RR has more forwarding overhead |
| **Cloud % Execution** | 2-4% | 4-8% | RR reaches cloud more often |

---

## Why inference.py vs finetunehighload.py

| Aspect | inference.py | finetunehighload.py |
|--------|--------------|------------------|
| **Purpose** | Evaluate finetuned model | Continue training on high load |
| **Modifies weights?** | ❌ No (greedy policy) | ✓ Yes (training) |
| **Good for ablation?** | ✅ Perfect | ❌ Wrong goal |
| **Reproducible?** | ✅ Yes (epsilon=0) | ❌ Weights change |
| **Supports sample_services?** | ✓ With modification | ✓ Native support |
| **Metric reliability** | ✓ Clean comparison | ❌ Training noise |

---

## Summary

**To evaluate EDF vs Round Robin on 3-2 tree with 10000 requests:**

1. **Generate requests** (only if 3-2 doesn't exist):
   ```bash
   python generate_requests_3_3.py --levels 3 --children 2 --requests 10000 --output training/data/sample_requests_10000_3_2.csv
   ```

2. **Run EDF** (config: SCHEDULING_ALGORITHM = "EDF"):
   ```bash
   python training/inference.py --checkpoint <checkpoint> --episodes 5 --requests 10000
   ```

3. **Run Round Robin** (config: SCHEDULING_ALGORITHM = "ROUND_ROBIN"):
   ```bash
   python training/inference.py --checkpoint <checkpoint> --episodes 5 --requests 10000
   ```

4. **Compare CSV outputs** for hit_rate, dropped count, latency metrics

**Use `inference.py`** - it's the right tool for clean, reproducible ablation studies.
