from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

try:
    from .parser import ALL_PARSER_STEPS
    from .resnet import build_arg_parser, run_cross_validation
except ImportError:
    from parser import ALL_PARSER_STEPS
    from resnet import build_arg_parser, run_cross_validation


ABLATIONS: dict[str, list[str]] = {
    "none": [],
    "crop": ["crop"],
    "crop_normalize": ["crop", "normalize"],
    "crop_normalize_rotate": ["crop", "normalize", "rotate"],
    "crop_normalize_rotate_jitter": ["crop", "normalize", "rotate", "jitter"],
    "full": list(ALL_PARSER_STEPS),
}


def main() -> None:
    parent = build_arg_parser()
    parent.description = "Run parser ablations with Messidor ResNet cross validation"
    parent.add_argument("--ablation-output-root", default="output/ablations")
    args = parent.parse_args()
    root = Path(args.ablation_output_root)
    summaries: dict[str, object] = {}
    for name, steps in ABLATIONS.items():
        run_args = copy.copy(args)
        run_args.parser_steps = steps
        run_args.output_dir = str(root / name)
        summaries[name] = run_cross_validation(run_args)
    root.mkdir(parents=True, exist_ok=True)
    (root / "ablation_results.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
