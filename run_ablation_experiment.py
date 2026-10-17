"""
run_ablation_experiment.py
---------------------------
Ablation study runner: removes the Double-DQN component from the framework
by swapping the RL strategy from DDQN (default) to vanilla DQN, runs both
configurations on the sample data, and produces a results table called
`AbalationExperiment` (saved as results/AbalationExperiment.csv and .md).

What is being ablated
----------------------
Baseline  : DDQN   -- online net selects the greedy next action, target net
                       evaluates it (Double Q-learning, reduces overestimation
                       bias). This is the framework's default RL strategy.
Ablation  : DQN    -- vanilla DQN target y = r + gamma * max_a' Q_target(s',a').
                       Action-selection/evaluation decoupling is REMOVED.

Everything else (topology, service catalogue, cache capacity, PER replay
buffer, network architecture, reward function, hyper-parameters) is kept
IDENTICAL between runs so that the Double-Q component is the only varying
factor.

Usage
-----
    python run_ablation_experiment.py
    python run_ablation_experiment.py --episodes 50 --requests 2000
    python run_ablation_experiment.py --agents ddqn dqn --seed 42

Output
------
    results/AbalationExperiment.csv   <- machine-readable table
    results/AbalationExperiment.md    <- human-readable markdown table
"""

import argparse
import os
import sys
import time
import csv
import statistics

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from training.train import train  # noqa: E402

RESULTS_DIR = os.path.join(_ROOT, "results")
TABLE_NAME  = "AbalationExperiment"


def _run_one(agent_type: str, episodes: int, req_per_ep: int, seed: int,
             services_csv: str) -> dict:
    """Run one training configuration and return summary metrics."""
    print("\n" + "#" * 72)
    print(f"#  ABLATION RUN  ->  RL strategy = {agent_type.upper()}")
    print("#" * 72 + "\n")

    checkpoint_path = os.path.join(_ROOT, "checkpoints", f"ablation_{agent_type}.pt")
    weights_path    = os.path.join(_ROOT, "checkpoints", f"ablation_{agent_type}_weights.pt")
    metrics_path    = os.path.join(RESULTS_DIR, f"ablation_{agent_type}_metrics.csv")

    t0 = time.time()
    orchestrator = train(
        num_episodes    = episodes,
        req_per_ep      = req_per_ep,
        seed            = seed,
        checkpoint_path = checkpoint_path,
        weights_path    = weights_path,
        metrics_path    = metrics_path,
        save_plots      = False,
        print_every     = max(1, episodes // 5),
        services_csv    = services_csv,
        agent_type      = agent_type,
    )
    elapsed = time.time() - t0

    rewards    = orchestrator.episode_rewards
    hit_rates  = orchestrator.episode_hit_rates
    profits    = orchestrator.episode_profits
    losses     = orchestrator.episode_losses
    violations = orchestrator.episode_violations
    overest_gaps=getattr(orchestrator,"episode_overestimation_gap",[])

    n_tail = max(1, len(rewards) // 5)  # last 20% of episodes = "converged" window

    def _avg(seq):
        return sum(seq) / len(seq) if seq else 0.0

    def _std(seq):
        return round(statistics.pstdev(seq),4) if len(seq) >1 else 0.0

    return {
        "agent_type"          : agent_type.upper(),
        "episodes"            : episodes,
        "requests_per_episode": req_per_ep,
        "avg_reward_last_20pct": round(_avg(rewards[-n_tail:]), 4),
        "avg_hit_rate_last_20pct": round(_avg(hit_rates[-n_tail:]) * 100, 2),
        "avg_profit_last_20pct": round(_avg(profits[-n_tail:]), 4),
        "std_reward_last_20pct": _std(rewards[-n_tail:]),
        "std_hit_rate_last_20pct":_std([h*100 for h in hit_rates[-n_tail:]]),
        "std_profit_last_20pct":_std(profits[-n_tail:]),
        "avg_overestimation_gap_last_20pct": round(_avg(overest_gaps[-n_tail:]),4),
        "final_avg_loss"      : round(losses[-1], 6) if losses else 0.0,
        "final_violations"    : violations[-1] if violations else 0,
        "final_epsilon"       : round(orchestrator.agent.epsilon, 4),
        "train_time_sec"      : round(elapsed, 2),
    }


def _write_table(rows: list, out_dir: str, table_name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{table_name}.csv")
    md_path  = os.path.join(out_dir, f"{table_name}.md")

    fields = list(rows[0].keys())

    # ---- CSV ------------------------------------------------------------
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # ---- Markdown ---------------------------------------------------------
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {table_name}\n\n")
        f.write(
            "Ablation study: RL strategy swapped from DDQN (baseline) to "
            "vanilla DQN (component removed = Double Q-learning decoupling "
            "of action selection/evaluation). All other framework "
            "components (topology, PER buffer, reward function, "
            "hyper-parameters) kept identical.\n\n"
        )
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("|" + "|".join(["---"] * len(fields)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[k]) for k in fields) + " |\n")

    print(f"\nAblation table written to:\n  {csv_path}\n  {md_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run an ablation experiment: DDQN (baseline) vs DQN "
                    "(Double-Q component removed) on the sample data."
    )
    p.add_argument("--agents", nargs="+", default=["ddqn", "dqn"],
                    choices=["ddqn", "dqn"],
                    help="Which RL strategies to run (default: both).")
    p.add_argument("--episodes", type=int, default=50,
                    help="Number of training episodes per run (default: 50, "
                         "kept small for a quick ablation pass).")
    p.add_argument("--requests", type=int, default=2000,
                    help="Requests per episode (default: 2000, sample-sized).")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed shared across all runs for fair comparison.")
    p.add_argument("--services", type=str,
                    default=os.path.join(_ROOT, "training", "data", "sample_services.csv"),
                    help="Path to the sample service catalogue CSV.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    services_csv = args.services if os.path.exists(args.services) else None
    if services_csv is None:
        print(f"[WARN] Sample services CSV not found at {args.services}; "
              f"falling back to an auto-generated catalogue.")

    rows = []
    for agent_type in args.agents:
        row = _run_one(
            agent_type   = agent_type,
            episodes     = args.episodes,
            req_per_ep   = args.requests,
            seed         = args.seed,
            services_csv = services_csv,
        )
        rows.append(row)

    print("\n" + "=" * 72)
    print("  ABLATION EXPERIMENT SUMMARY")
    print("=" * 72)
    for row in rows:
        print(row)

    _write_table(rows, RESULTS_DIR, TABLE_NAME)


if __name__ == "__main__":
    main()
