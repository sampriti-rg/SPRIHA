"""
run_pipeline.py
---------------
Single-command full training pipeline for the Edge Caching DDQN agent.

Handles all five phases in the correct sequence:

  Phase 0  -- Regenerate sample_requests.csv for the current topology
  Phase 1  -- Train DDQN from scratch on the service catalogue
  Phase 2  -- Collect realistic transition data (epsilon=1.0 exploration)
  Phase 3  -- Fine-tune the trained model with realistic data
  Phase 4  -- Evaluate DDQN vs Baseline on the sample request trace

When to use this script
-----------------------
  - Topology changed  (NUM_LEVELS, NUM_CHILDREN)   -> full pipeline required
  - Cache size changed (NODE_CACHE_CAP)             -> Phase 1 onwards
  - Reward formula changed                          -> Phase 1 onwards
  - Service catalogue changed only                  -> Phase 2 onwards (--from-collect)

Usage
-----
  # Full pipeline (topology or reward formula changed)
  python run_pipeline.py

  # Use a specific service catalogue
  python run_pipeline.py --services training/data/sample_services.csv

  # Skip Phase 0 (requests already match topology)
  python run_pipeline.py --skip-requests

  # Start from Phase 2 collect (topology unchanged, only services changed)
  python run_pipeline.py --from-collect

  # Custom episode counts
  python run_pipeline.py --train-episodes 500 --collect-episodes 50 --finetune-episodes 150

  # Skip plots for faster CI runs
  python run_pipeline.py --no-plots

Checkpoints written
-------------------
  checkpoints/ddqn_final.pt          after Phase 1 (train)
  checkpoints/ddqn_finetuned.pt      after Phase 3 (finetune)

Results written
---------------
  results/training_metrics.csv
  results/finetune_metrics.csv
  results/custom_eval_report.txt
  results/custom_eval_requests.csv
  training/data/sample_requests.csv  (regenerated in Phase 0)
  training/data/collected_*.csv      (regenerated in Phase 2)
"""

import argparse
import datetime
import os
import sys
import time

# Ensure project root is on path
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from edge_caching.config import (
    NUM_LEVELS, NUM_CHILDREN, NODE_CACHE_CAP,
    NUM_EPISODES, REQUESTS_PER_EPISODE, PROVISIONING_INTERVAL,
    RANDOM_SEED, TRAFFIC_MODEL, RESULTS_DIR,
)

_SEP  = "=" * 72
_DIV  = "-" * 72
_STAR = "*" * 72

_DATA_DIR        = os.path.join(_ROOT, "training", "data")
_CHECKPOINT_DIR  = os.path.join(_ROOT, "checkpoints")
_REQUESTS_CSV    = os.path.join(_DATA_DIR, "sample_requests.csv")
_SERVICES_CSV    = os.path.join(_DATA_DIR, "sample_services.csv")
_CHECKPOINT_INIT = os.path.join(_CHECKPOINT_DIR, "ddqn_final.pt")
_CHECKPOINT_FINE = os.path.join(_CHECKPOINT_DIR, "ddqn_finetuned.pt")


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------

def _banner(phase: int, title: str, config_summary: str) -> None:
    print()
    print(_STAR)
    print(f"  PHASE {phase}  --  {title}")
    print(f"  Config: {config_summary}")
    print(f"  Time  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(_STAR)


def _done(phase: int, elapsed: float) -> None:
    print()
    print(_DIV)
    print(f"  Phase {phase} complete  ({elapsed:.1f}s)")
    print(_DIV)


# ---------------------------------------------------------------------------
# Phase 0 -- Regenerate sample_requests.csv
# ---------------------------------------------------------------------------

def phase0_regenerate_requests(n: int, services_csv: str, seed: int) -> None:
    """
    Regenerate sample_requests.csv using leaf nodes derived from the
    current config topology (NUM_LEVELS, NUM_CHILDREN).

    This must run whenever the topology changes so that origin_node_id
    values in the CSV match nodes that actually exist in the network.
    """
    _banner(0, "Regenerate Request Trace", 
            f"L={NUM_LEVELS}, C={NUM_CHILDREN}, cap={NODE_CACHE_CAP}, n={n}")

    t0 = time.time()
    from training.data.generate_requests import generate
    generate(n=n, seed=seed, out=_REQUESTS_CSV, svc_csv=services_csv)
    _done(0, time.time() - t0)


# ---------------------------------------------------------------------------
# Phase 1 -- Train from scratch
# ---------------------------------------------------------------------------

def phase1_train(
    num_episodes  : int,
    services_csv  : str,
    seed          : int,
    traffic_model : str,
    save_plots    : bool,
    print_every   : int,
) -> None:
    """
    Full DDQN training from random weights.
    Required when topology, reward formula, or STATE_DIM changes.
    Saves: checkpoints/ddqn_final.pt
    """
    _banner(1, "Train DDQN from Scratch",
            f"L={NUM_LEVELS}, C={NUM_CHILDREN}, cap={NODE_CACHE_CAP}, "
            f"episodes={num_episodes}")

    t0 = time.time()
    from training.train import train
    train(
        num_episodes    = num_episodes,
        seed            = seed,
        checkpoint_path = _CHECKPOINT_INIT,
        save_plots      = save_plots,
        traffic_model   = traffic_model,
        print_every     = print_every,
        services_csv    = services_csv if os.path.exists(services_csv) else None,
    )
    _done(1, time.time() - t0)


# ---------------------------------------------------------------------------
# Phase 2 -- Collect realistic transition data
# ---------------------------------------------------------------------------

def phase2_collect(
    num_episodes  : int,
    services_csv  : str,
    seed          : int,
    traffic_model : str,
) -> None:
    """
    Run exploration simulation (epsilon=1.0) with realistic service catalogue.
    Records (s, a, r, s', done) transitions for fine-tuning.
    Saves: training/data/collected_*.csv
    """
    _banner(2, "Collect Realistic Transition Data",
            f"episodes={num_episodes}, services={os.path.basename(services_csv)}")

    t0 = time.time()
    from training.collect_stats import collect
    collect(
        services_csv = services_csv,
        num_episodes = num_episodes,
        seed         = seed,
        traffic_model= traffic_model,
    )
    _done(2, time.time() - t0)


# ---------------------------------------------------------------------------
# Phase 3 -- Fine-tune
# ---------------------------------------------------------------------------

def phase3_finetune(
    num_episodes  : int,
    seed          : int,
    traffic_model : str,
    save_plots    : bool,
    print_every   : int,
) -> None:
    """
    Load ddqn_final.pt, pre-fill buffer with collected transitions,
    continue training with realistic service data.
    Saves: checkpoints/ddqn_finetuned.pt
    """
    _banner(3, "Fine-Tune with Realistic Data",
            f"episodes={num_episodes}, base={os.path.basename(_CHECKPOINT_INIT)}")

    t0 = time.time()
    from training.finetune import finetune
    finetune(
        num_episodes  = num_episodes,
        seed          = seed,
        start_fresh   = False,
        save_plots    = save_plots,
        traffic_model = traffic_model,
        print_every   = print_every,
    )
    _done(3, time.time() - t0)


# ---------------------------------------------------------------------------
# Phase 4 -- Evaluate
# ---------------------------------------------------------------------------

def phase4_evaluate(services_csv: str, seed: int) -> None:
    """
    Run BASELINE vs DDQN evaluation on the regenerated request trace.
    Saves: results/custom_eval_report.txt + results/custom_eval_requests.csv
    """
    _banner(4, "Evaluate DDQN vs Baseline",
            f"requests={os.path.basename(_REQUESTS_CSV)}, "
            f"checkpoint={os.path.basename(_CHECKPOINT_FINE)}")

    t0 = time.time()
    from training.evaluate_custom_requests import evaluate
    evaluate(
        requests_csv    = _REQUESTS_CSV,
        services_csv    = services_csv if os.path.exists(services_csv) else None,
        checkpoint_path = _CHECKPOINT_FINE,
        seed            = seed,
        save_outputs    = True,
    )
    _done(4, time.time() - t0)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    pipeline_start = time.time()

    print()
    print(_SEP)
    print("  EDGE CACHING DDQN  --  FULL TRAINING PIPELINE")
    print(f"  Started : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(_SEP)
    print(f"  Topology        : {NUM_LEVELS} levels, {NUM_CHILDREN} children")
    print(f"  Cache capacity  : {NODE_CACHE_CAP} slots per node")
    print(f"  Services CSV    : {args.services}")
    print(f"  Train episodes  : {args.train_episodes}")
    print(f"  Collect episodes: {args.collect_episodes}")
    print(f"  Finetune episodes:{args.finetune_episodes}")
    print(f"  Request count   : {args.num_requests}")
    print(f"  Traffic model   : {args.traffic}")
    print(f"  Seed            : {args.seed}")
    print()
    print("  Phases to run:")
    if not args.skip_requests and not args.from_collect:
        print("    [0] Regenerate sample_requests.csv  (topology-aware)")
    if not args.from_collect:
        print("    [1] Train DDQN from scratch")
    print("    [2] Collect realistic transition data")
    print("    [3] Fine-tune trained model")
    print("    [4] Evaluate DDQN vs Baseline")
    print(_SEP)

    # Ensure output directories exist
    os.makedirs(_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.join(_ROOT, RESULTS_DIR), exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 0 -- Regenerate requests
    # ------------------------------------------------------------------
    if not args.skip_requests and not args.from_collect:
        phase0_regenerate_requests(
            n            = args.num_requests,
            services_csv = args.services,
            seed         = args.seed,
        )
    else:
        print(f"\n  [Phase 0 skipped]  Using existing: {_REQUESTS_CSV}")

    # ------------------------------------------------------------------
    # Phase 1 -- Train from scratch
    # ------------------------------------------------------------------
    if not args.from_collect:
        phase1_train(
            num_episodes  = args.train_episodes,
            services_csv  = args.services,
            seed          = args.seed,
            traffic_model = args.traffic,
            save_plots    = not args.no_plots,
            print_every   = args.print_every,
        )
    else:
        print(f"\n  [Phase 1 skipped]  Using checkpoint: {_CHECKPOINT_INIT}")
        if not os.path.exists(_CHECKPOINT_INIT):
            print(f"\n  ERROR: --from-collect requires {_CHECKPOINT_INIT} to exist.")
            print(f"         Run without --from-collect first to train the model.")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 2 -- Collect
    # ------------------------------------------------------------------
    phase2_collect(
        num_episodes  = args.collect_episodes,
        services_csv  = args.services,
        seed          = args.seed,
        traffic_model = args.traffic,
    )

    # ------------------------------------------------------------------
    # Phase 3 -- Fine-tune
    # ------------------------------------------------------------------
    phase3_finetune(
        num_episodes  = args.finetune_episodes,
        seed          = args.seed,
        traffic_model = args.traffic,
        save_plots    = not args.no_plots,
        print_every   = args.print_every,
    )

    # ------------------------------------------------------------------
    # Phase 4 -- Evaluate
    # ------------------------------------------------------------------
    phase4_evaluate(
        services_csv = args.services,
        seed         = args.seed,
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total = time.time() - pipeline_start
    mins, secs = divmod(int(total), 60)

    print()
    print(_STAR)
    print("  PIPELINE COMPLETE")
    print(f"  Total time      : {mins}m {secs}s")
    print(f"  Checkpoint      : {_CHECKPOINT_FINE}")
    print(f"  Eval report     : {os.path.join(_ROOT, RESULTS_DIR, 'custom_eval_report.txt')}")
    print(f"  Training metrics: {os.path.join(_ROOT, RESULTS_DIR, 'training_metrics.csv')}")
    print(f"  Finetune metrics: {os.path.join(_ROOT, RESULTS_DIR, 'finetune_metrics.csv')}")
    print(_STAR)
    print()
    print("  Next steps:")
    print("    Inspect results/custom_eval_report.txt for DDQN vs Baseline comparison")
    print("    Run: python main.py simulate   to inspect a single operational episode")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Full DDQN training pipeline: requests -> train -> collect -> finetune -> evaluate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Full pipeline (topology or reward changed)\n"
            "  python run_pipeline.py\n\n"
            "  # With specific services\n"
            "  python run_pipeline.py --services training/data/sample_services.csv\n\n"
            "  # Skip request regeneration (topology unchanged)\n"
            "  python run_pipeline.py --skip-requests\n\n"
            "  # Only collect + finetune + evaluate (topology + train unchanged)\n"
            "  python run_pipeline.py --from-collect\n\n"
            "  # Custom episode counts\n"
            "  python run_pipeline.py --train-episodes 500 --collect-episodes 50 --finetune-episodes 150\n"
        ),
    )

    p.add_argument(
        "--services", type=str, default=_SERVICES_CSV,
        help=f"Service catalogue CSV (default: {_SERVICES_CSV})",
    )
    p.add_argument(
        "--num-requests", type=int, default=5000,
        help="Number of requests to generate in Phase 0 (default: 5000)",
    )
    p.add_argument(
        "--train-episodes", type=int, default=NUM_EPISODES,
        help=f"Phase 1 training episodes (default: {NUM_EPISODES})",
    )
    p.add_argument(
        "--collect-episodes", type=int, default=50,
        help="Phase 2 collection episodes (default: 50)",
    )
    p.add_argument(
        "--finetune-episodes", type=int, default=150,
        help="Phase 3 fine-tuning episodes (default: 150)",
    )
    p.add_argument(
        "--traffic", type=str, default=TRAFFIC_MODEL,
        choices=["poisson_zipf", "uniform"],
        help=f"Traffic model (default: {TRAFFIC_MODEL})",
    )
    p.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})",
    )
    p.add_argument(
        "--skip-requests", action="store_true",
        help="Skip Phase 0: do not regenerate sample_requests.csv",
    )
    p.add_argument(
        "--from-collect", action="store_true",
        help=(
            "Skip Phases 0 and 1 (requests + train). "
            "Start from Phase 2 (collect). "
            "Requires checkpoints/ddqn_final.pt to exist."
        ),
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help="Skip saving visualisation plots",
    )
    p.add_argument(
        "--print-every", type=int, default=10,
        help="Print episode summary every N episodes (default: 10)",
    )
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()
    run_pipeline(args)
