"""Compute FR3/Wuji statistics using the same composed experiment as training."""

import argparse

from scripts.run_training import preflight
from scripts.run_training import resolve_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("experiment")
    parser.add_argument("--stats-batch-size", type=int, default=512)
    args, overrides = parser.parse_known_args()
    config = resolve_experiment(args.experiment, overrides)
    preflight(config, require_norm=False)
    from examples.fr3_wuji.compute_norm_stats import compute

    compute(config, batch_size=args.stats_batch_size)


if __name__ == "__main__":
    main()
