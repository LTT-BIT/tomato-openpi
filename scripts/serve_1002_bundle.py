"""Serve the exported 1002 checkpoint using only bundle-relative assets."""

import argparse
import dataclasses
import logging
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    os.environ["OPENPI_DATA_HOME"] = str(root / "cache")
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config

    checkpoint = root / "checkpoint"
    cfg = config.get_config("pi05_fr3_wuji_1002_subtasks_detailed_30hz")
    cfg = dataclasses.replace(
        cfg,
        data=dataclasses.replace(
            cfg.data,
            assets=config.AssetsConfig(assets_dir=str(checkpoint / "assets"), asset_id="1002_subtasks_detailed"),
        ),
    )
    policy = policy_config.create_trained_policy(cfg, checkpoint)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
