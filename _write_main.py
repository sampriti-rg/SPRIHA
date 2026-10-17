"""Helper: writes main.py as a clean UTF-8 file."""

src = '''\
"""
main.py
-------
Single entry point for the Edge Caching DDQN project.

Dispatches to one of five commands:

    train       Phase 0 -- Train DDQN from scratch on simulated data
    collect     Phase 1 -- Run Poisson+Zipf simulation, collect statistics
                           and write training data files to training/data/
    finetune    Phase 2 -- Fine-tune trained model with realistic collected data
    inference   Phase 3 -- Run trained model with greedy policy (epsilon=0)
    simulate    Operational single-episode simulation runner (run_simulation.py)

Usage
-----
    python main.py train      [--episodes N] [--traffic poisson_zipf|uniform]
    python main.py collect    [--episodes N] [--services PATH]
    python main.py finetune   [--episodes N] [--fresh]
    python main.py inference  [--checkpoint PATH] [--episodes N]
    python main.py simulate
    python main.py --help
"""

import argparse
import os
import sys


# ---------------------------------------------------------------------------
# Command dispatchers
# ---------------------------------------------------------------------------

def _run_train(args):
    from training.train import train
    kwargs = dict(seed=args.seed, save_plots=not args.no_plots,
                  print_every=args.print_every, traffic_model=args.traffic,
                  num_episodes=args.episodes)
    if args.checkpoint:
        kwargs["checkpoint_path"] = args.checkpoint
    train(**kwargs)


def _run_collect(args):
    from training.collect_stats import collect
    collect(services_csv=args.services, num_episodes=args.episodes,
            req_per_ep=args.requests, seed=args.seed,
            traffic_model=args.traffic)


def _run_finetune(args):
    from training.finetune import finetune
    finetune(num_episodes=args.episodes, seed=args.seed,
             start_fresh=args.fresh, save_plots=not args.no_plots,
             print_every=args.print_every, traffic_model=args.traffic)


def _run_inference(args):
    from training.inference import provision_demo
    provision_demo(checkpoint_path=args.checkpoint, num_episodes=args.episodes,
                   req_per_ep=args.requests, seed=args.seed,
                   save_csv=not args.no_csv, traffic_model=args.traffic)


def _run_simulate(args):
    from run_simulation import main as sim_main
    sim_main()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser():
    root = argparse.ArgumentParser(
        prog="main.py",
        description="Edge Caching DDQN -- single entry point.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\\n"
            "  python main.py train\\n"
            "  python main.py train --episodes 500 --traffic poisson_zipf\\n"
            "  python main.py collect --episodes 30\\n"
            "  python main.py collect --episodes 50 --traffic uniform\\n"
            "  python main.py finetune --episodes 100\\n"
            "  python main.py finetune --episodes 150 --fresh\\n"
            "  python main.py inference --checkpoint checkpoints/ddqn_finetuned.pt\\n"
            "  python main.py simulate\\n"
        ),
    )
    sub = root.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    def _add_common(p):
        p.add_argument("--episodes",  type=int, default=None,
                       help="Number of episodes (overrides config default)")
        p.add_argument("--requests",  type=int, default=None,
                       help="Requests per episode (overrides config default)")
        p.add_argument("--seed",      type=int, default=42,
                       help="Random seed (default: 42)")
        p.add_argument("--traffic",   type=str, default="poisson_zipf",
                       choices=["poisson_zipf", "uniform"],
                       help="Traffic model: poisson_zipf (default) or uniform")

    # train
    p_train = sub.add_parser(
        "train",
        help="Phase 0: Train DDQN from scratch on randomly generated simulated data",
        description=(
            "Train the DDQN agent from scratch using Poisson+Zipf traffic\\n"
            "and a randomly generated service catalogue.\\n"
            "Saves checkpoint to checkpoints/ddqn_final.pt"
        ),
    )
    _add_common(p_train)
    p_train.add_argument("--checkpoint",  type=str, default=None,
                         help="Override output checkpoint path")
    p_train.add_argument("--no-plots",    action="store_true",
                         help="Skip saving visualisation plots")
    p_train.add_argument("--print-every", type=int, default=10,
                         help="Print summary every N episodes (default: 10)")
    p_train.set_defaults(func=_run_train)

    # collect
    p_collect = sub.add_parser(
        "collect",
        help="Phase 1: Simulate Poisson+Zipf traffic and write statistics to training/data/",
        description=(
            "Run a warm-up simulation with realistic service data.\\n"
            "Records per-interval statistics, normalised state vectors, and\\n"
            "full (s,a,r,s prime,done) transitions to training/data/.\\n\\n"
            "Files written:\\n"
            "  training/data/collected_stats.csv       raw statistics\\n"
            "  training/data/collected_states.csv      8-element state vectors\\n"
            "  training/data/collected_transitions.csv transitions\\n\\n"
            "Agent uses epsilon=1.0 so INSERT/DELETE/KEEP are sampled uniformly."
        ),
    )
    _add_common(p_collect)
    p_collect.add_argument(
        "--services", type=str,
        default=os.path.join("training", "data", "sample_services.csv"),
        help="Path to service definitions CSV (default: training/data/sample_services.csv)",
    )
    p_collect.set_defaults(func=_run_collect)

    # finetune
    p_fine = sub.add_parser(
        "finetune",
        help="Phase 2: Fine-tune trained model using collected realistic statistics",
        description=(
            "Load checkpoints/ddqn_final.pt, pre-fill the replay buffer with\\n"
            "realistic transitions from training/data/collected_transitions.csv,\\n"
            "then continue training with the realistic service catalogue.\\n"
            "Saves fine-tuned checkpoint to checkpoints/ddqn_finetuned.pt"
        ),
    )
    _add_common(p_fine)
    p_fine.add_argument("--fresh",       action="store_true",
                        help="Start from fresh weights instead of pre-trained checkpoint")
    p_fine.add_argument("--no-plots",    action="store_true",
                        help="Skip saving visualisation plots")
    p_fine.add_argument("--print-every", type=int, default=10,
                        help="Print summary every N episodes (default: 10)")
    p_fine.set_defaults(func=_run_finetune)

    # inference
    p_infer = sub.add_parser(
        "inference",
        help="Phase 3: Run a trained model with greedy policy (epsilon=0)",
        description=(
            "Load a trained checkpoint and run the simulation with epsilon=0\\n"
            "(fully greedy policy, no exploration).\\n"
            "Saves per-request CSV and summary report to results/."
        ),
    )
    _add_common(p_infer)
    p_infer.add_argument(
        "--checkpoint", type=str,
        default=os.path.join("checkpoints", "ddqn_final.pt"),
        help="Checkpoint to load (default: checkpoints/ddqn_final.pt)",
    )
    p_infer.add_argument("--no-csv", action="store_true",
                         help="Skip saving CSV and report to results/")
    p_infer.set_defaults(func=_run_inference)

    # simulate
    p_sim = sub.add_parser(
        "simulate",
        help="Run the single-episode operational simulation (run_simulation.py)",
        description=(
            "Run one episode of the edge caching simulation and produce\\n"
            "a detailed per-request log and summary report in results/."
        ),
    )
    p_sim.set_defaults(func=_run_simulate)

    return root


def _apply_defaults(args):
    from edge_caching.config import NUM_EPISODES, REQUESTS_PER_EPISODE
    if getattr(args, "episodes", None) is None:
        args.episodes = NUM_EPISODES
    if getattr(args, "requests", None) is None:
        args.requests = REQUESTS_PER_EPISODE


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = _build_parser()
    args   = parser.parse_args()
    _apply_defaults(args)

    _SEP = "=" * 72
    labels = {
        "train"    : "Phase 0  --  Initial Training",
        "collect"  : "Phase 1  --  Statistics Collection",
        "finetune" : "Phase 2  --  Fine-Tuning",
        "inference": "Phase 3  --  Inference",
        "simulate" : "Operational Simulation",
    }
    print(_SEP)
    print(f"  Edge Caching DDQN  |  {labels.get(args.command, args.command)}")
    print(_SEP)
    print()

    args.func(args)


if __name__ == "__main__":
    main()
'''

with open("main.py", "w", encoding="utf-8") as f:
    f.write(src)

print("main.py written successfully.")
