# -*- coding: utf-8 -*-
"""
compare_policies.py
-------------------
Side-by-side performance comparison of three cache provisioning policies
run against the same request trace.

Policies
--------
  DDQN     : Trained model (checkpoint), epsilon=0 pure greedy policy.
             The Q-network observes a 10-element state vector per
             (service, node) pair and outputs the learned INSERT / DELETE /
             KEEP action.  No further learning occurs during evaluation
             (replay buffer starts empty so is_ready() stays False).

  Greedy   : Demand-reactive heuristic -- no learning, no Q-network.
             INSERT  when a service is NOT cached AND had cache misses
                     (forwarded_count > 0) this interval.
             DELETE  when a service IS  cached AND was completely idle
                     (executed=0, served=0, forwarded=0) this interval.
             KEEP    in all other cases.
             Approximates LFU eviction + demand-driven insertion.

  Baseline : Static cache -- priority_init warm-start then KEEP forever.
             Represents a "deploy once, never adapt" operator strategy.
             Any metric advantage over this line is the direct value of
             dynamic cache management.

All three policies share identical request routing via EdgeNode.receive_request().
The only difference is what action is chosen at each provisioning interval.

Metrics
-------
  hit_rate_mean       Rolling (window=50) hit rate mean
  hit_rate_std        Rolling hit rate std-dev (stability)
  local_rate          Requests served at origin leaf / total
  cloud_rate          Requests escalated to cloud / total
  drop_rate           Deadline violations / total
  total_revenue       sum(revenue * omega) for all executed requests
  rev_*               Revenue split by criticality tier
  rev_per_request     Revenue per successfully executed request
  executed / dropped  Raw counts

Usage
-----
  python compare_policies.py

  python compare_policies.py \\
      --checkpoint checkpoints/ddqn_finetune_v4.pt \\
      --steps 100 500 1000 2000 5000 10000

  python compare_policies.py \\
      --checkpoint checkpoints/ddqn_finetuned.pt \\
      --requests   training/data/sample_requests.csv \\
      --services   training/data/sample_services.csv \\
      --steps      100 250 500 1000 2000 5000 10000 \\
      --out-csv    results/policy_comparison.csv \\
      --out-plot   results/policy_comparison.png

Outputs
-------
  results/policy_comparison.csv   All three policies x all load levels
  results/policy_comparison.png   6-panel comparison dashboard


Execution Command
python compare_policies_All.py --metric local --out-plot results/local_exec.png
python compare_policies_All.py --metric cloud --out-plot results/cloud_esc.png
python compare_policies_All.py --metric drop --out-plot results/drop_rate.png
python compare_policies_All.py --metric revenue --out-plot results/revenue.png
"""

import argparse
import csv
import os
import sys
import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from edge_caching.config import (
    NUM_LEVELS, NUM_CHILDREN, NODE_CACHE_CAP,
    PROVISIONING_INTERVAL, RANDOM_SEED, RESULTS_DIR,
)
from edge_caching.core.node import EdgeNode, ACTION_INSERT, ACTION_DELETE, ACTION_KEEP
from edge_caching.core.service import Service, generate_service_catalogue
from edge_caching.core.request import Request
from edge_caching.network.topology import NetworkTopology
from edge_caching.control.orchestrator import EdgeOrchestrator
from edge_caching.agents.ddqn_agent import DDQNAgent
from edge_caching.utils.simulation.request_generator import load_requests_from_csv
from edge_caching.policies.top_down_placement import make_top_down_policy
from edge_caching.policies.lca_placement import make_lca_policy

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_CHECKPOINT = os.path.join(_ROOT, "checkpoints", "ddqn_finetuned_resumed_500epv3.pt")
_DEFAULT_REQUESTS   = os.path.join(_ROOT, "training", "data", "sample_requests_10000.csv")
_DEFAULT_SERVICES   = os.path.join(_ROOT, "training", "data", "sample_services.csv")
_DEFAULT_OUT_CSV    = os.path.join(_ROOT, RESULTS_DIR, "policy_comparison.csv")
_DEFAULT_OUT_PLOT   = os.path.join(_ROOT, RESULTS_DIR, "policy_comparison.png")
_DEFAULT_STEPS      = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

_TIERS    = ["mission_critical", "high", "standard", "best_effort"]
_POLICIES = ["DDQN", "LCA", "TopDown", "Greedy"]

_SEP = "=" * 80
_DIV = "-" * 80

_CSV_COLUMNS = [
    "policy", "n_requests",
    "hit_rate_mean", "hit_rate_std",
    "local_rate", "cloud_rate", "drop_rate",
    "total_revenue",
    "rev_mission_critical", "rev_high", "rev_standard", "rev_best_effort",
    "rev_per_request", "executed", "dropped",
]

# Callable type: (node, service, snapshot) -> action int (0/1/2)
ActionFn = Callable[[EdgeNode, Service, dict], int]


# ---------------------------------------------------------------------------
# Service catalogue loader
# ---------------------------------------------------------------------------

def _load_services(csv_path: str) -> Dict[int, Service]:
    catalogue: Dict[int, Service] = {}
    with open(csv_path, newline="", encoding="utf-8-sig", errors="replace") as f:
        for row in csv.DictReader(f):
            svc = Service(
                service_id      = int(row["service_id"]),
                size            = int(row["size"]),
                popularity      = float(row["popularity"]),
                criticality     = row["criticality"].strip(),
                revenue         = float(row["revenue"]),
                init_cpu_cycles = float(row["init_cpu_cycles"])
                                  if "init_cpu_cycles" in row and row["init_cpu_cycles"]
                                  else 50_000,
            )
            catalogue[svc.service_id] = svc
    return catalogue


# ---------------------------------------------------------------------------
# World reset -- identical starting state for every (policy, load-level) run
# ---------------------------------------------------------------------------

def _reset_world(topology: NetworkTopology, catalogue: Dict[int, Service]) -> None:
    """
    Wipe all node caches, EDF queues, counters, and revenue accumulators,
    then re-seed with priority_init so every run begins from the same state.
    """
    for node in [topology.cloud_root] + topology.all_nodes:
        node.cache.clear()
        node.cache_sizes.clear()
        node._queue.clear()
        node.total_revenue = 0.0
        node.reset_interval_stats()
    for svc in catalogue.values():
        svc.reset_interval_counters()
    topology.prime_cloud_cache(catalogue)
    topology.priority_init(catalogue)


# ---------------------------------------------------------------------------
# Execution-tier classifier
# ---------------------------------------------------------------------------

def _classify_tier(req: Request) -> str:
    """Classify where a completed request was ultimately handled."""
    if req.status == "dropped":
        return "dropped"
    if req.executed_at_node is None:
        return "unknown"
    if req.executed_at_node == req.origin_node_id:
        return "leaf"
    if req.executed_at_node == "CLOUD":
        return "cloud"
    for _, evt in req.node_log:
        if evt == "cache_miss_peer":
            return "peer"
        if evt in ("cache_miss_parent", "cache_miss_cloud"):
            return "parent"
    return "parent"


# ---------------------------------------------------------------------------
# Policy action functions
# ---------------------------------------------------------------------------

def make_ddqn_policy(agent: DDQNAgent) -> ActionFn:
    """
    Return a closure over the trained DDQN agent.

    At each provision cycle the 10-element state vector is built from
    (service, node snapshot) and fed through the online Q-network.
    argmax over the three Q-values gives the action (epsilon=0).
    """
    def _policy(node: EdgeNode, service: Service, snap: dict) -> int:
        state = DDQNAgent.build_state_vector(service, snap)
        return agent.select_action(state)
    return _policy


def greedy_policy(node: EdgeNode, service: Service, snap: dict) -> int:
    """
    Greedy heuristic -- demand-reactive, no learning.

    Logic (evaluated once per (node, service) per provision interval):

      INSERT : service is NOT in cache AND had forwarded (missed) requests
               this interval.  The node is actively turning away traffic it
               could serve locally -- insert to capture that demand next time.

      DELETE : service IS in cache AND was completely idle (no hits, no
               misses, nothing served).  Cache space is wasted -- evict to
               make room for something that might be requested.

      KEEP   : everything else -- don't disturb a working or ambiguous state.

    This approximates LFU eviction + miss-driven insertion without any
    Q-network or historical state -- the purest reactive heuristic.
    """
    svc_id    = service.service_id
    fwd       = snap["forwarded_count"].get(svc_id, 0)   # cache misses
    exe       = snap["executed_count"].get(svc_id, 0)    # local hits
    srv       = snap["served_count"].get(svc_id, 0)      # served (pre-exec)
    is_cached = svc_id in snap["cached_services"]

    if not is_cached and fwd > 0:
        return ACTION_INSERT
    if is_cached and exe == 0 and srv == 0 and fwd == 0:
        return ACTION_DELETE
    return ACTION_KEEP


def static_policy(node: EdgeNode, service: Service, snap: dict) -> int:
    """
    Static baseline -- always KEEP.

    The cache is seeded once by priority_init and never modified.
    Any metric advantage shown by DDQN or Greedy over this line is the
    direct value of dynamic provisioning.
    """  # noqa: RUF001
    return ACTION_KEEP


# ---------------------------------------------------------------------------
# Generic policy runner (Greedy + Baseline)
# ---------------------------------------------------------------------------

def _run_heuristic_policy(
    requests      : List[Request],
    topology      : NetworkTopology,
    catalogue     : Dict[int, Service],
    policy_fn     : ActionFn,
    prov_interval : int,
    policy_name   : str,
) -> dict:
    """
    Evaluate a heuristic policy (Greedy or Baseline) on a fixed request trace.

    Fast process  : node.receive_request() -- identical to DDQN runner.
    Slow process  : lightweight provision loop that:
                    1. Snapshots every node (get_stats_snapshot)
                    2. Calls policy_fn(node, service, snap) -> action
                    3. Applies action via node.apply_action()
                    4. Updates service popularity EMA
                    5. Resets interval stats

    No Q-network, replay buffer, or gradient updates are involved.
    """
    _reset_world(topology, catalogue)

    tiers           = defaultdict(int)
    revenue_total   = 0.0
    revenue_by_tier = {t: 0.0 for t in _TIERS}
    executed_flags  : List[int] = []
    total           = len(requests)

    for idx, req in enumerate(requests):
        # ?? Track request origin for policy-specific tracking ??????????
        if hasattr(policy_fn, 'lca_instance'):
            policy_fn.lca_instance.track_request(req.service_id, req.origin_node_id)
        if hasattr(policy_fn, 'td_instance'):
            policy_fn.td_instance.track_request(req.service_id, req.origin_node_id)

        # ?? Fast process: route the request ?????????????????????????
        topology.node_map[req.origin_node_id].receive_request(req, catalogue)

        tier = _classify_tier(req)
        tiers[tier] += 1
        executed_flags.append(1 if req.status == "executed" else 0)

        if req.status == "executed":
            svc = catalogue[req.service_id]
            rv  = svc.revenue * svc.omega
            revenue_total                    += rv
            revenue_by_tier[svc.criticality] += rv

        # ?? Slow process: provision every N requests ?????????????????
        if (idx + 1) % prov_interval == 0:
            # Top-Down needs to compute all decisions first
            if hasattr(policy_fn, 'td_instance'):
                policy_fn.td_instance.compute_all_decisions(catalogue)

            for node in topology.all_nodes:
                snap = node.get_stats_snapshot()
                for svc_id, service in catalogue.items():
                    action = policy_fn(node, service, snap)
                    node.apply_action(action, service)

            # Update popularity EMA and reset interval counters
            total_demand = max(
                sum(s.demand_count for s in catalogue.values()), 1
            )
            for service in catalogue.values():
                service.update_popularity(total_demand)
                service.reset_interval_counters()
            for node in topology.all_nodes:
                node.reset_interval_stats()

            # Reset policy tracking for next interval
            if hasattr(policy_fn, 'lca_instance'):
                policy_fn.lca_instance.reset_interval()
            if hasattr(policy_fn, 'td_instance'):
                policy_fn.td_instance.reset_interval()

    return _build_metrics(
        policy_name, total, executed_flags, tiers,
        revenue_total, revenue_by_tier,
    )


# ---------------------------------------------------------------------------
# DDQN policy runner (reuses EdgeOrchestrator.provision())
# ---------------------------------------------------------------------------

def _run_ddqn_policy(
    requests      : List[Request],
    topology      : NetworkTopology,
    catalogue     : Dict[int, Service],
    orchestrator  : EdgeOrchestrator,
    prov_interval : int,
) -> dict:
    """
    Evaluate the DDQN policy using EdgeOrchestrator.provision().

    epsilon is set to 0 before calling so all actions are greedy argmax.
    The replay buffer starts empty so memory.is_ready() stays False and
    no gradient updates occur -- this is pure inference.

    The orchestrator's internal _prev_snapshots and _last_provision_sim_time
    are reset at the start of each run so state doesn't bleed between load
    levels.
    """
    _reset_world(topology, catalogue)

    # Reset orchestrator state for a clean evaluation run
    orchestrator._prev_snapshots           = {}
    orchestrator._last_provision_sim_time  = 0.0
    orchestrator.agent.epsilon             = 0.0

    tiers           = defaultdict(int)
    revenue_total   = 0.0
    revenue_by_tier = {t: 0.0 for t in _TIERS}
    executed_flags  : List[int] = []
    total           = len(requests)
    prov_calls      = 0
    total_prov      = max(total // prov_interval, 1)

    for idx, req in enumerate(requests):
        # ?? Fast process ?????????????????????????????????????????????
        topology.node_map[req.origin_node_id].receive_request(req, catalogue)

        tier = _classify_tier(req)
        tiers[tier] += 1
        executed_flags.append(1 if req.status == "executed" else 0)

        if req.status == "executed":
            svc = catalogue[req.service_id]
            rv  = svc.revenue * svc.omega
            revenue_total                    += rv
            revenue_by_tier[svc.criticality] += rv

        # ?? Slow process ?????????????????????????????????????????????
        if (idx + 1) % prov_interval == 0:
            prov_calls += 1
            orchestrator.provision(
                episode_done = (prov_calls == total_prov),
                sim_time_ms  = req.arrival_time,
            )

    # Finalise episode counters (epsilon decay etc.) -- harmless for eval
    executed = sum(executed_flags)
    dropped  = tiers["dropped"]
    orchestrator.end_episode(
        episode_reward   = 0.0,
        total_executed   = executed,
        total_requests   = total,
        total_forwarded  = tiers["peer"] + tiers["parent"] + tiers["cloud"],
        total_violations = dropped,
        avg_loss         = 0.0,
    )

    return _build_metrics(
        "DDQN", total, executed_flags, tiers,
        revenue_total, revenue_by_tier,
    )


# ---------------------------------------------------------------------------
# Shared metric builder
# ---------------------------------------------------------------------------

def _build_metrics(
    policy_name     : str,
    total           : int,
    executed_flags  : List[int],
    tiers           : dict,
    revenue_total   : float,
    revenue_by_tier : dict,
) -> dict:
    """Compute rolling hit-rate stats and package all metrics into a dict."""
    executed = sum(executed_flags)
    dropped  = tiers["dropped"]

    # Rolling hit rate (window = 50 requests)
    window    = 50
    flags_arr = np.array(executed_flags, dtype=float)
    if len(flags_arr) >= window:
        rolling = np.convolve(flags_arr, np.ones(window) / window, mode="valid")
        hr_mean = float(rolling.mean())
        hr_std  = float(rolling.std())
    else:
        hr_mean = float(flags_arr.mean()) if len(flags_arr) > 0 else 0.0
        hr_std  = 0.0

    return {
        "policy"               : policy_name,
        "n_requests"           : total,
        "hit_rate_mean"        : round(hr_mean,                                         6),
        "hit_rate_std"         : round(hr_std,                                          6),
        "local_rate"           : round(tiers["leaf"]  / max(total, 1),                 6),
        "cloud_rate"           : round(tiers["cloud"] / max(total, 1),                 6),
        "drop_rate"            : round(dropped        / max(total, 1),                 6),
        "total_revenue"        : round(revenue_total,                                   4),
        "rev_mission_critical" : round(revenue_by_tier.get("mission_critical", 0.0),   4),
        "rev_high"             : round(revenue_by_tier.get("high",             0.0),   4),
        "rev_standard"         : round(revenue_by_tier.get("standard",         0.0),   4),
        "rev_best_effort"      : round(revenue_by_tier.get("best_effort",      0.0),   4),
        "rev_per_request"      : round(revenue_total / max(executed, 1),               6),
        "executed"             : executed,
        "dropped"              : dropped,
    }


# ---------------------------------------------------------------------------
# 6-panel comparison plot
# ---------------------------------------------------------------------------

def _plot_comparison(
    results  : Dict[str, List[dict]],
    out_path : str,
    metric   : str = "all",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    _COLOURS = {"DDQN": "#1f77b4", "TopDown": "#ff7f0e", "LCA": "#2ca02c", "Greedy": "#d62728"}
    _MARKERS  = {"DDQN": "o",       "TopDown": "s",      "LCA": "D",       "Greedy": "^"}
    _LS       = {"DDQN": "-",       "TopDown": "--",     "LCA": "-.",      "Greedy": ":"}

    # Define all available metrics
    all_metrics = {
        "local": {
            "col": "local_rate",
            "title": "Local Execution Rate\n(served at origin leaf node)",
            "ylabel": "Local Rate", "pct": True, "better": "higher",
        },
        "cloud": {
            "col": "cloud_rate",
            "title": "Cloud Escalation Rate\n(requests reaching cloud fallback)",
            "ylabel": "Cloud Rate", "pct": True, "better": "lower",
        },
        "drop": {
            "col": "drop_rate",
            "title": "Drop Rate\n(SLA / deadline violations)",
            "ylabel": "Drop Rate", "pct": True, "better": "lower",
        },
        "revenue": {
            "col": "total_revenue",
            "title": "Total Revenue\n(sum of revenue for executed requests)",
            "ylabel": "Revenue", "pct": False, "better": "higher",
        },
    }

    # Select metrics based on parameter
    if metric == "all":
        panels = list(all_metrics.values())
    else:
        if metric not in all_metrics:
            print(f"  WARNING: Unknown metric '{metric}', using 'all'")
            panels = list(all_metrics.values())
        else:
            panels = [all_metrics[metric]]

    # ------------------------------------------------------------------
    # Compute shared x-tick positions
    # ------------------------------------------------------------------
    all_x_vals: set = set()
    for rows in results.values():
        for r in rows:
            all_x_vals.add(r["n_requests"])
    max_x = max(all_x_vals) if all_x_vals else 0

    anchor_ticks = {0}
    if max_x >= 100:
        anchor_ticks.add(100)
    if max_x >= 500:
        anchor_ticks.add(500)
    x_ticks = sorted(anchor_ticks | all_x_vals)

    # Determine subplot layout
    if len(panels) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes = [axes]  # Make it iterable
    elif len(panels) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes = axes.flatten()
    elif len(panels) == 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes = axes.flatten()
    else:  # 4 panels
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

    ckpt_name = os.path.basename(_DEFAULT_CHECKPOINT)
    metric_name = metric.upper() if metric != "all" else "ALL METRICS"
    fig.suptitle(
        f"Cache Provisioning Policy Comparison ({metric_name})\n"
        f"DDQN ({ckpt_name})  vs  All Policies",
        fontsize=13, fontweight="bold", y=0.98 if len(panels) > 2 else 1.02,
    )

    for ax, cfg in zip(axes, panels):
        col = cfg["col"]
        for policy_name in _POLICIES:
            rows = results.get(policy_name, [])
            if not rows:
                continue
            xs = [r["n_requests"] for r in rows]
            ys = [r[col]          for r in rows]
            ax.plot(
                xs, ys,
                color=_COLOURS.get(policy_name, "#000000"),
                marker=_MARKERS.get(policy_name, "o"),
                linestyle=_LS.get(policy_name, "-"),
                linewidth=2.5,
                markersize=8,
                label=policy_name,
                zorder=3,
            )
            # Annotate last data point
            if xs:
                ann = f"{ys[-1]*100:.1f}%" if cfg["pct"] else f"{ys[-1]:,.0f}"
                ax.annotate(
                    ann,
                    xy=(float(xs[-1]), float(ys[-1])),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                    color=_COLOURS.get(policy_name, "#000000"),
                    fontweight="bold",
                )

        # Y-axis: force origin at 0; 8% headroom for annotations
        current_top = ax.get_ylim()[1]
        ax.set_ylim(bottom=0, top=current_top * 1.08)

        # X-axis: force origin at 0
        ax.set_xlim(left=0, right=max_x * 1.05)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(
            [str(v) for v in x_ticks],
            rotation=45,
            ha="right",
            fontsize=8,
        )

        if cfg["pct"]:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

        direction = "(higher is better)" if cfg["better"] == "higher" else "(lower is better)"
        ax.set_title(cfg["title"], fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Number of Requests", fontsize=10, fontweight="bold")
        ax.set_ylabel(cfg["ylabel"], fontsize=10, fontweight="bold")
        ax.legend(fontsize=10, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.tick_params(axis="y", labelsize=9)
        ax.text(
            0.02, 0.98, direction,
            transform=ax.transAxes,
            fontsize=8, color="grey", va="top", style="italic",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96] if len(panels) > 2 else None)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved  -> {out_path}")


# ---------------------------------------------------------------------------
# Console summary helpers
# ---------------------------------------------------------------------------

def _print_per_policy_table(results: Dict[str, List[dict]], steps: List[int]) -> None:
    """Print a per-load-level table for each policy."""
    print(
        f"\n  {'Policy':<10}  {'N':>6}  "
        f"{'HitMean%':>9}  {'HitStd%':>8}  "
        f"{'Local%':>7}  {'Cloud%':>7}  {'Drop%':>6}  "
        f"{'Revenue':>10}  {'Rev/Req':>8}"
    )
    print(_DIV)
    for policy_name in _POLICIES:
        for m in results.get(policy_name, []):
            print(
                f"  {policy_name:<10}  {m['n_requests']:>6}  "
                f"{m['hit_rate_mean']*100:>8.2f}%  "
                f"{m['hit_rate_std']*100:>7.2f}%  "
                f"{m['local_rate']*100:>6.2f}%  "
                f"{m['cloud_rate']*100:>6.2f}%  "
                f"{m['drop_rate']*100:>5.2f}%  "
                f"{m['total_revenue']:>10.2f}  "
                f"{m['rev_per_request']:>8.4f}"
            )
        print()


def _print_summary_table(results: Dict[str, List[dict]], steps: List[int]) -> None:
    """Print DDQN deltas vs Top-Down at the maximum load level."""
    print(_SEP)
    print(f"  SUMMARY AT MAXIMUM LOAD  (N = {steps[-1]})")
    print(f"  DDQN delta shows improvement (+) or regression (-) vs Top-Down baseline")
    print(_DIV)

    col_metrics = [
        ("Local exec rate",     "local_rate",           True),
        ("Cloud escalation",    "cloud_rate",           True),
        ("Drop rate",           "drop_rate",            True),
        ("Total revenue",       "total_revenue",        False),
    ]

    last = {p: results[p][-1] for p in _POLICIES if results.get(p)}

    # Print header with all policies
    header = f"  {'Metric':<24}"
    for p in _POLICIES:
        if p in last:
            header += f"  {p:>10}"
    print(header)
    print(_DIV)

    for label, col, pct in col_metrics:
        row = f"  {label:<24}"
        for p in _POLICIES:
            if p in last:
                val = last[p][col]
                if pct:
                    row += f"  {val*100:>9.2f}%"
                else:
                    row += f"  {val:>10,.2f}"
        print(row)
    print(_SEP)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    checkpoint    : str       = _DEFAULT_CHECKPOINT,
    requests_csv  : str       = _DEFAULT_REQUESTS,
    services_csv  : str       = _DEFAULT_SERVICES,
    steps         : List[int] = _DEFAULT_STEPS,
    prov_interval : int       = PROVISIONING_INTERVAL,
    seed          : int       = RANDOM_SEED,
    metric        : str       = "all",
    out_csv       : str       = _DEFAULT_OUT_CSV,
    out_plot      : str       = _DEFAULT_OUT_PLOT,
) -> None:

    random.seed(seed)
    np.random.seed(seed)

    print(_SEP)
    print("  Policy Comparison: DDQN vs LCA vs TopDown vs Greedy")
    print(_DIV)
    print(f"  Checkpoint    : {checkpoint}")
    print(f"  Requests      : {requests_csv}")
    print(f"  Services      : {services_csv}")
    print(f"  Load steps    : {steps}")
    print(f"  Prov interval : {prov_interval}")
    print(f"  Metric        : {metric}")
    print(f"  Output CSV    : {out_csv}")
    print(f"  Output plot   : {out_plot}")
    print(_SEP)

    # ?? Build service catalogue ???????????????????????????????????????????
    if services_csv and os.path.exists(services_csv):
        catalogue = _load_services(services_csv)
        print(f"  Loaded {len(catalogue)} services from {services_csv}")
    else:
        catalogue = {
            s.service_id: s
            for s in generate_service_catalogue(50, seed=seed)
        }
        print(f"  Generated {len(catalogue)} random services (no CSV found)")

    # ?? Build topology ????????????????????????????????????????????????????
    topology = NetworkTopology(
        num_levels   = NUM_LEVELS,
        num_children = NUM_CHILDREN,
        cache_cap    = NODE_CACHE_CAP,
    )
    topology.prime_cloud_cache(catalogue)
    topology.priority_init(catalogue)
    print(f"  Topology      : {NUM_LEVELS} levels, {NUM_CHILDREN} children, "
          f"{len(topology.all_nodes)} edge nodes, "
          f"{len(topology.leaf_nodes)} leaf nodes")

    # ?? Load request trace ????????????????????????????????????????????????
    if not os.path.exists(requests_csv):
        print(f"\n  ERROR: request trace not found: {requests_csv}")
        print("  Generate one with:  python generate_request_data.py")
        sys.exit(1)

    all_requests = load_requests_from_csv(requests_csv, catalogue, topology.node_map)
    n_total = len(all_requests)
    print(f"  Loaded        : {n_total} requests")

    # Clamp steps to the available request count
    steps = sorted(set(min(n, n_total) for n in steps))
    print(f"  Effective steps: {steps}\n")

    # ?? Build DDQN orchestrator (inference only, epsilon=0) ???????????????
    orchestrator = EdgeOrchestrator(nodes=topology.all_nodes, catalogue=catalogue)
    if os.path.exists(checkpoint):
        orchestrator.load_checkpoint(checkpoint)
        orchestrator.agent.epsilon = 0.0
        print(f"  DDQN checkpoint loaded  ->  epsilon=0 (greedy policy)")
    else:
        orchestrator.agent.epsilon = 0.0
        print(f"  WARNING: checkpoint not found at {checkpoint}")
        print(f"           Running DDQN with untrained (random-init) weights")
    print()

    # Policy registry
    # make_ddqn_policy wraps the agent so select_action() feeds the Q-network
    ddqn_action_fn : ActionFn = make_ddqn_policy(orchestrator.agent)

    # Create LCA policy instance
    lca_action_fn : ActionFn = make_lca_policy(topology)

    # Create Top-Down policy instance
    topdown_action_fn : ActionFn = make_top_down_policy(topology)

    # Greedy policy (function, no instantiation needed)
    greedy_action_fn : ActionFn = greedy_policy

    # Run all (policy x load-level) combinations
    results   : Dict[str, List[dict]] = {p: [] for p in _POLICIES}
    all_rows  : List[dict]             = []

    for policy_name in _POLICIES:
        print(f"  Running: {policy_name}")
        print(_DIV)

        for n in steps:
            subset = all_requests[:n]

            if policy_name == "DDQN":
                m = _run_ddqn_policy(
                    requests      = subset,
                    topology      = topology,
                    catalogue     = catalogue,
                    orchestrator  = orchestrator,
                    prov_interval = prov_interval,
                )
            elif policy_name == "LCA":
                m = _run_heuristic_policy(
                    requests      = subset,
                    topology      = topology,
                    catalogue     = catalogue,
                    policy_fn     = lca_action_fn,
                    prov_interval = prov_interval,
                    policy_name   = "LCA",
                )
            elif policy_name == "TopDown":
                m = _run_heuristic_policy(
                    requests      = subset,
                    topology      = topology,
                    catalogue     = catalogue,
                    policy_fn     = topdown_action_fn,
                    prov_interval = prov_interval,
                    policy_name   = "TopDown",
                )
            elif policy_name == "Greedy":
                m = _run_heuristic_policy(
                    requests      = subset,
                    topology      = topology,
                    catalogue     = catalogue,
                    policy_fn     = greedy_action_fn,
                    prov_interval = prov_interval,
                    policy_name   = "Greedy",
                )
            else:
                raise ValueError(f"Unknown policy: {policy_name}")

            results[policy_name].append(m)
            all_rows.append(m)

            print(
                f"    N={n:>6}  "
                f"Hit={m['hit_rate_mean']*100:>5.1f}%  "
                f"Local={m['local_rate']*100:>5.1f}%  "
                f"Cloud={m['cloud_rate']*100:>5.1f}%  "
                f"Drop={m['drop_rate']*100:>4.1f}%  "
                f"Rev={m['total_revenue']:>9.2f}  "
                f"Exec={m['executed']:>6}  Drop={m['dropped']:>5}"
            )
        print()

    # ?? Write CSV ??????????????????????????????????????????????????????????
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Results CSV saved -> {out_csv}")

    # ?? Plot comparison ??????????????????????????????????????????????
    _plot_comparison(results, out_plot, metric)

    # Per-policy breakdown table
    _print_per_policy_table(results, steps)

    # Delta summary at max load
    _print_summary_table(results, steps)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="compare_policies.py",
        description="Compare DDQN vs Greedy vs Static Baseline cache provisioning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python compare_policies.py\n\n"
            "  python compare_policies.py \\\n"
            "      --checkpoint checkpoints/ddqn_finetune_v4.pt\n\n"
            "  python compare_policies.py \\\n"
            "      --checkpoint checkpoints/ddqn_finetuned.pt \\\n"
            "      --steps 100 500 1000 5000 10000 \\\n"
            "      --out-csv results/policy_comparison.csv \\\n"
            "      --out-plot results/policy_comparison.png\n"
        ),
    )
    p.add_argument(
        "--checkpoint", type=str, default=_DEFAULT_CHECKPOINT,
        help=f"DDQN checkpoint .pt file (default: {_DEFAULT_CHECKPOINT})",
    )
    p.add_argument(
        "--requests", type=str, default=_DEFAULT_REQUESTS,
        help=f"Request trace CSV (default: {_DEFAULT_REQUESTS})",
    )
    p.add_argument(
        "--services", type=str, default=_DEFAULT_SERVICES,
        help=f"Service catalogue CSV (default: {_DEFAULT_SERVICES})",
    )
    p.add_argument(
        "--steps", type=int, nargs="+", default=_DEFAULT_STEPS,
        help="Request load levels to evaluate (default: 100 250 500 1000 2000 5000 10000)",
    )
    p.add_argument(
        "--prov", type=int, default=PROVISIONING_INTERVAL,
        help=f"Provisioning interval in requests (default: {PROVISIONING_INTERVAL})",
    )
    p.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})",
    )
    p.add_argument(
        "--metric", type=str, 
        choices=["local", "cloud", "drop", "revenue", "all"],
        default="all",
        help="Which metric to plot: local (execution rate), cloud (escalation rate), drop (rate), revenue, or all (default: all)",
    )
    p.add_argument(
        "--out-csv", type=str, default=_DEFAULT_OUT_CSV,
        help=f"Output CSV path (default: {_DEFAULT_OUT_CSV})",
    )
    p.add_argument(
        "--out-plot", type=str, default=_DEFAULT_OUT_PLOT,
        help=f"Output PNG path (default: {_DEFAULT_OUT_PLOT})",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        checkpoint    = args.checkpoint,
        requests_csv  = args.requests,
        services_csv  = args.services,
        steps         = sorted(set(args.steps)),
        prov_interval = args.prov,
        seed          = args.seed,
        metric        = args.metric,
        out_csv       = args.out_csv,
        out_plot      = args.out_plot,
    )
