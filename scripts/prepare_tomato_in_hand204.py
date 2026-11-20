"""Freeze and materialize a NEW full204 five-phase four-RGB B training bundle."""

import argparse
import json
from pathlib import Path

from openpi.subtasks import tomato_in_hand as contract


def materialize(selection, output, tokenizer):
    """Historical fixed release wrapper; new datasets use prepare_tomato_in_hand."""
    if (
        selection.get("version") != contract.VERSION
        or len(selection["episodes"]) != contract.EPISODES
        or selection["frames"] != contract.FRAMES
    ):
        raise ValueError("Verified full204 frozen selection required")
    from scripts.prepare_tomato_in_hand import materialize as prepare

    return prepare(selection, output, tokenizer)


def main():
    from openpi.subtasks.prepare import write_json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=contract.DEFAULT_SOURCE)
    p.add_argument("--out", type=Path, default=contract.DEFAULT_BUNDLE)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists() or a.selection.exists():
        raise FileExistsError("Use new output paths")
    frozen = contract.freeze(a.source)
    write_json(a.selection, frozen)
    import sentencepiece

    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(a.tokenizer))
    result = materialize(frozen, a.out, tokenizer)
    print(json.dumps({k: result[k] for k in ["episodes", "frames", "token_budget"]}, indent=2))


if __name__ == "__main__":
    main()
