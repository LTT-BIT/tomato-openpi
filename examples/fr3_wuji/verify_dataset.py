"""Compatibility import for historical commands; use scripts.verify_dataset."""

from scripts.verify_dataset import *  # noqa: F403

if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))  # noqa: F405
