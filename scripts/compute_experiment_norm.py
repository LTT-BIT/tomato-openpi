"""Compute FR3/Wuji or FR3/Sharpa statistics from the same experiment as training."""

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
    custom_compute = getattr(config.data, "compute_normalization", None)
    if custom_compute is not None:
        custom_compute(config, batch_size=args.stats_batch_size)
        return
    from examples.fr3_wuji.compute_norm_stats import compute

    compute(config, batch_size=args.stats_batch_size)


if __name__ == "__main__":
    main()
