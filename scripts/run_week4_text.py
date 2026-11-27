"""Run the Week 4 baseline, EDGAR pilot, corpus or incremental tests."""

import argparse
from importlib import import_module
from pathlib import Path

from ingen_qr.week4 import baseline
from ingen_qr.week4.common import TEXT_CONFIG, config, require_baseline, stage_lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["baseline", "pilot", "full", "incremental"])
    parser.add_argument("--config", type=Path, default=TEXT_CONFIG)
    parser.add_argument("--week3-predictions", type=Path)
    args = parser.parse_args()
    settings = config(args.config)
    root = Path(settings["output_directory"])
    if args.stage == "baseline":
        baseline.verify(root, args.week3_predictions)
        return
    gate = require_baseline(root)
    with stage_lock(root / "text_stage_lock.db"):
        if args.stage == "incremental":
            import_module("ingen_qr.week4.incremental").run(settings, root, gate)
        else:
            import_module("ingen_qr.week4.text_pipeline").run(
                settings, root, gate, args.stage
            )


if __name__ == "__main__":
    main()
