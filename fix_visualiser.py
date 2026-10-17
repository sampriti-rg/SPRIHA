# -*- coding: utf-8 -*-
"""Write a clean visualiser.py with plot_forwarded_requests correctly defined."""
import os

path = os.path.join("edge_caching", "utils", "visualiser.py")
os.makedirs(os.path.dirname(path), exist_ok=True)

code = '''\
"""
visualiser.py - Generates result graphs after the simulation completes.
hit_rate panels show leaf-only local hit rate (Fix 4 corrected metric).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List
from edge_caching.config import RESULTS_DIR


def _smooth(data, window=10):
    arr = np.array(data, dtype=float)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def _ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def plot_episode_reward(rewards):
    out = _ensure_dir()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.3, color="steelblue", label="Raw")
    ax.plot(_smooth(rewards), color="steelblue", linewidth=2, label="Smoothed")
    ax.set_title("Episode Reward (DDQN Convergence)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "episode_reward.png"), dpi=150)
    plt.close(fig)


def plot_cumulative_profit(profits):
    out = _ensure_dir()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.cumsum(profits), color="green", linewidth=2)
    ax.set_title("Cumulative Profit Over Episodes")
    ax.set_xlabel("Episode"); ax.set_ylabel("Cumulative Revenue")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "cumulative_profit.png"), dpi=150)
    plt.close(fig)


def plot_cache_hit_rate(hit_rates):
    out = _ensure_dir()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hit_rates, alpha=0.3, color="darkorange", label="Raw")
    ax.plot(_smooth(hit_rates), color="darkorange", linewidth=2, label="Smoothed")
    ax.set_title("Local Hit Rate per Episode (Leaf-Only)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Local Hit Rate")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "cache_hit_rate.png"), dpi=150)
    plt.close(fig)


def plot_forwarded_requests(fwd_counts):
    out = _ensure_dir()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(fwd_counts, alpha=0.3, color="tomato", label="Raw")
    ax.plot(_smooth(fwd_counts), color="tomato", linewidth=2, label="Smoothed")
    ax.set_title("Forwarded Requests per Episode")
    ax.set_xlabel("Episode"); ax.set_ylabel("Forwarded Count")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "forwarded_requests.png"), dpi=150)
    plt.close(fig)


def plot_training_loss(losses):
    out = _ensure_dir()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, alpha=0.3, color="purple", label="Raw")
    ax.plot(_smooth(losses), color="purple", linewidth=2, label="Smoothed")
    ax.set_title("DDQN Training Loss per Episode")
    ax.set_xlabel("Episode"); ax.set_ylabel("Huber Loss")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "training_loss.png"), dpi=150)
    plt.close(fig)


def plot_sla_violations(violations):
    out = _ensure_dir()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(violations, alpha=0.3, color="crimson", label="Raw")
    ax.plot(_smooth(violations), color="crimson", linewidth=2, label="Smoothed")
    ax.set_title("SLA Violations per Episode")
    ax.set_xlabel("Episode"); ax.set_ylabel("Violation Count")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sla_violations.png"), dpi=150)
    plt.close(fig)


def plot_node_cache_utilization(nodes):
    out = _ensure_dir()
    node_ids  = [n.node_id for n in nodes]
    util_vals = [n.cache_utilization() for n in nodes]
    fig, ax = plt.subplots(figsize=(max(8, len(nodes) * 0.6), 4))
    bars = ax.bar(node_ids, util_vals, color="teal", edgecolor="white")
    ax.set_title("Final Cache Utilization per Node")
    ax.set_xlabel("Node"); ax.set_ylabel("Utilization (0-1)")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Full capacity")
    ax.legend(); plt.xticks(rotation=45, ha="right")
    for bar, val in zip(bars, util_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "node_cache_utilization.png"), dpi=150)
    plt.close(fig)


def plot_dashboard(rewards, profits, hit_rates, fwd_counts, losses, nodes, violations=None):
    out = _ensure_dir()
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Edge Caching DDQN - Training Dashboard", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(rewards, alpha=0.25, color="steelblue")
    ax1.plot(_smooth(rewards), color="steelblue", linewidth=2)
    ax1.set_title("Episode Reward"); ax1.set_xlabel("Episode"); ax1.set_ylabel("Reward")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(np.cumsum(profits), color="green", linewidth=2)
    ax2.set_title("Cumulative Profit"); ax2.set_xlabel("Episode"); ax2.set_ylabel("Revenue")
    ax2.grid(True, linestyle="--", alpha=0.4)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(hit_rates, alpha=0.25, color="darkorange")
    ax3.plot(_smooth(hit_rates), color="darkorange", linewidth=2)
    ax3.set_title("Local Hit Rate (Leaf-Only)")
    ax3.set_xlabel("Episode"); ax3.set_ylabel("Local Hit Rate"); ax3.set_ylim(0, 1.05)
    ax3.grid(True, linestyle="--", alpha=0.4)

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(fwd_counts, alpha=0.25, color="tomato")
    ax4.plot(_smooth(fwd_counts), color="tomato", linewidth=2)
    ax4.set_title("Forwarded Requests"); ax4.set_xlabel("Episode"); ax4.set_ylabel("Count")
    ax4.grid(True, linestyle="--", alpha=0.4)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(losses, alpha=0.25, color="purple")
    ax5.plot(_smooth(losses), color="purple", linewidth=2)
    ax5.set_title("Training Loss (Huber)"); ax5.set_xlabel("Episode"); ax5.set_ylabel("Loss")
    ax5.grid(True, linestyle="--", alpha=0.4)

    ax6 = fig.add_subplot(gs[1, 2])
    if violations:
        ax6.plot(violations, alpha=0.25, color="crimson")
        ax6.plot(_smooth(violations), color="crimson", linewidth=2)
    ax6.set_title("SLA Violations"); ax6.set_xlabel("Episode"); ax6.set_ylabel("Dropped")
    ax6.grid(True, linestyle="--", alpha=0.4)

    ax7 = fig.add_subplot(gs[2, :])
    node_ids  = [n.node_id for n in nodes]
    util_vals = [n.cache_utilization() for n in nodes]
    bars = ax7.bar(node_ids, util_vals, color="teal", edgecolor="white")
    ax7.set_title("Final Cache Utilization per Node")
    ax7.set_xlabel("Node"); ax7.set_ylabel("Utilization"); ax7.set_ylim(0, 1.1)
    ax7.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, util_vals):
        ax7.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    plt.setp(ax7.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.savefig(os.path.join(out, "dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Dashboard saved -> {os.path.join(out, 'dashboard.png')}")


def save_all_results(orchestrator, topology):
    o = orchestrator; n = topology.all_nodes
    plot_episode_reward(o.episode_rewards)
    plot_cumulative_profit(o.episode_profits)
    plot_cache_hit_rate(o.episode_hit_rates)
    plot_forwarded_requests(o.episode_fwd_counts)
    plot_training_loss(o.episode_losses)
    plot_sla_violations(o.episode_violations)
    plot_node_cache_utilization(n)
    plot_dashboard(
        rewards=o.episode_rewards, profits=o.episode_profits,
        hit_rates=o.episode_hit_rates, fwd_counts=o.episode_fwd_counts,
        losses=o.episode_losses, violations=o.episode_violations, nodes=n,
    )
    print(f"All plots saved to: ./{RESULTS_DIR}/")
'''

with open(path, "w", encoding="utf-8") as f:
    f.write(code)
print(f"Written: {path}")

