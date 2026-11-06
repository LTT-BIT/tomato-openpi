"""JSONL decision worker for external phase-model probabilities; never robot control.

Request: {"id":1,"timestamp":0.5,"probabilities":[...seven numbers...]}
The future deployment client supplies causal PhaseModel.predict results.
This worker alone cannot recognize images and does not imply agent accuracy.
"""

import json
import sys

from openpi.subtasks.tomato import PhaseDecision


def main():
    decision = PhaseDecision()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("reset") is True:
                decision.reset()
                response = {"reset": True}
            else:
                response = decision.update(request["probabilities"], request["timestamp"])
            print(json.dumps({"id": request.get("id"), **response}, allow_nan=False), flush=True)
        except (KeyError, TypeError, ValueError) as exc:
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
