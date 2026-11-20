"""Compatibility import for historical commands; use scripts.compute_robot_norm_stats."""

from scripts.compute_robot_norm_stats import *  # noqa: F403

if __name__ == "__main__":
    import tyro

    tyro.cli(main)  # noqa: F405
