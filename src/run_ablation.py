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


PREPROCESS_ABLATIONS: dict[str, list[str]] = {
    "none": [],
    "normalize": ["normalize"],
    "clahe": ["clahe"],
    "rotate": ["rotate"],
    "normalize_clahe": ["normalize", "clahe"],
    "normalize_rotate": ["normalize", "rotate"],
    "clahe_rotate": ["clahe", "rotate"],
    "full": list(ALL_PREPROCESS_STEPS),
}

IMBALANCE_ABLATIONS: dict[str, dict[str, object]] = {
    "ce_no_sampler": {"loss_type": "ce", "classifier": "linear", "drw_start_epoch": 0, "sampler_power": 0.0},
    "ce_sampler": {"loss_type": "ce", "classifier": "linear", "drw_start_epoch": 0, "sampler_power": 0.25},
    "ldam_only": {"loss_type": "ldam", "classifier": "normed", "drw_start_epoch": 0, "sampler_power": 0.0},
    "ldam_drw": {"loss_type": "ldam", "classifier": "normed", "drw_start_epoch": 31, "sampler_power": 0.0},
    "ldam_drw_sampler": {
        "loss_type": "ldam",
        "classifier": "normed",
        "drw_start_epoch": 31,
        "sampler_power": 0.25,
        "sampler_joint_weight": 0.25,
    },
}


def _run_case(args: argparse.Namespace, root: Path, name: str, steps: list[str],
              overrides: dict[str, object]) -> dict[str, object]:
    run_args = copy.copy(args)
    run_args.preprocess_steps = steps
    for key, value in overrides.items():
        setattr(run_args, key, value)
    run_args.output_dir = str(root / name)
    return run_cross_validation(run_args)


def main() -> None:
    parent = build_cli()
    parent.description = "Run preprocess and imbalance ablations with Messidor ResNet cross validation"
    parent.add_argument("--ablation-output-root", default=str(DEFAULT_OUTPUT_ROOT / "ablations"))
    parent.add_argument("--ablation-kind", default="preprocess", choices=["preprocess", "imbalance", "both"])
    args = parent.parse_args()
    root = Path(args.ablation_output_root).expanduser()
    summaries: dict[str, object] = {}

    if args.ablation_kind in {"preprocess", "both"}:
        for name, steps in PREPROCESS_ABLATIONS.items():
            case_name = name if args.ablation_kind == "preprocess" else f"preprocess/{name}"
            summaries[case_name] = _run_case(args, root, case_name, steps, {})

    if args.ablation_kind in {"imbalance", "both"}:
        for name, overrides in IMBALANCE_ABLATIONS.items():
            case_name = name if args.ablation_kind == "imbalance" else f"imbalance/{name}"
            summaries[case_name] = _run_case(args, root, case_name, list(ALL_PREPROCESS_STEPS), overrides)

    root.mkdir(parents=True, exist_ok=True)
    (root / "ablation_results.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
