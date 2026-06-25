from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

try:
    from .preprocess import ALL_PREPROCESS_STEPS
    from .resnet import DEFAULT_OUTPUT_ROOT, build_cli, run_cross_validation
except ImportError:
    from preprocess import ALL_PREPROCESS_STEPS
    from resnet import DEFAULT_OUTPUT_ROOT, build_cli, run_cross_validation


ABLATIONS: dict[str, list[str]] = {
    "none": [],
    "normalize": ["normalize"],
    "normalize_rotate": ["normalize", "rotate"],
    "full": list(ALL_PREPROCESS_STEPS),
}


def main() -> None:
    parent = build_cli()
    parent.description = "Run preprocess ablations with Messidor ResNet cross validation"
    parent.add_argument("--ablation-output-root", default=str(DEFAULT_OUTPUT_ROOT / "ablations"))
    args = parent.parse_args()
    root = Path(args.ablation_output_root).expanduser()
    summaries: dict[str, object] = {}
    for name, steps in ABLATIONS.items():
        run_args = copy.copy(args)
        run_args.preprocess_steps = steps
        run_args.output_dir = str(root / name)
        summaries[name] = run_cross_validation(run_args)
    root.mkdir(parents=True, exist_ok=True)
    (root / "ablation_results.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
