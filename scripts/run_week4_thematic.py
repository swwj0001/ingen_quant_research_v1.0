"""Prepare the thematic panel, diagnose its transfer gap and apply the selected change."""

import argparse
from pathlib import Path

from ingen_qr.week4.common import config, require_baseline, stage_lock
from ingen_qr.week4.thematic import run
from ingen_qr.week4.thematic_panel import build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "diagnose", "apply"])
    parser.add_argument(
        "--config", type=Path, default=Path("configs/week4_thematic.yaml")
    )
    args = parser.parse_args()
    settings = config(args.config)
    text = config(Path(settings["text_config"]))
    root = Path(text["output_directory"])
    gate = require_baseline(root)
    with stage_lock(root / "thematic_stage_lock.db"):
        if args.stage == "prepare":
            build(settings, text, root / "thematic")
        else:
            run(settings, root, gate, apply_remedy=args.stage == "apply")


if __name__ == "__main__":
    main()
