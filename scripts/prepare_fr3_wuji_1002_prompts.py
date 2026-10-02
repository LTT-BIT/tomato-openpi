"""Create the 1002 subtask dataset with reviewed English instructions."""

import argparse
import json
from pathlib import Path
import tempfile

from create_fr3_wuji_episode_subset import create_subset
from create_fr3_wuji_episode_subset import read_jsonl
from create_fr3_wuji_episode_subset import write_jsonl
from prepare_fr3_wuji_20hz import compute_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-file", type=Path, required=True)
    args = parser.parse_args()
    prompts = json.loads(args.prompts_file.read_text())
    assert set(prompts) == {"0", "1", "2", "3"}
    assert all(isinstance(p, str) and p.strip() for p in prompts.values())
    source = Path("/home/descfly/data-clean/subtasks")
    output = Path("/home/descfly/data-clean/1002_subtasks_detailed")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    episodes = read_jsonl(source / "meta/episodes.jsonl")
    assert len(episodes) == 444
    assert all(e["segment_id"] == f"subtask_{e['task_index'] + 1}" for e in episodes)
    with tempfile.TemporaryDirectory(prefix=".1002-prompts-", dir=output.parent) as temp:
        root = Path(temp) / "dataset"
        create_subset(source, root, [e["episode_index"] for e in episodes])
        tasks = read_jsonl(root / "meta/tasks.jsonl")
        for t in tasks:
            t["task"] = prompts[str(t["task_index"])]
        for e in episodes:
            e["tasks"] = [prompts[str(e["task_index"])]]
        write_jsonl(root / "meta/tasks.jsonl", tasks)
        write_jsonl(root / "meta/episodes.jsonl", episodes)
        compute_stats((root,), root, 256, action_hz=30.0, rewrite_prompt=False)
        provenance = json.loads((root / "provenance.json").read_text())
        provenance.update(
            datasets=[str(output)],
            source=str(source),
            selection="All 444 annotated subtasks, including complete tasks shorter than 50 frames",
            prompt_mapping=prompts,
        )
        (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        root.rename(output)
    print(f"Prepared {output}")


if __name__ == "__main__":
    main()
