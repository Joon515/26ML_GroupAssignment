from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt

try:
    from .parser import ParserConfig, load_rgb, parser_stages
    from .resnet import load_messidor_records
except ImportError:
    from parser import ParserConfig, load_rgb, parser_stages
    from resnet import load_messidor_records


def save_parser_samples(data_root: Path, output_dir: Path, image_size: int, sample_count: int, seed: int,
                        epochs: int) -> None:
    records = load_messidor_records(data_root)
    rng = random.Random(seed)
    selected = rng.sample(records, k=min(sample_count, len(records)))
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_sample in output_dir.glob("parser_sample_*.png"):
        old_sample.unlink()

    config = ParserConfig(image_size=image_size)
    for sample_index, record in enumerate(selected, start=1):
        image = load_rgb(record.image_path)
        stages = parser_stages(image, config, rng=random.Random(seed + epochs * 1_000_003 + sample_index))
        fig, axes = plt.subplots(1, len(stages), figsize=(3 * len(stages), 3))
        if len(stages) == 1:
            axes = [axes]
        for axis, (name, stage_image) in zip(axes, stages):
            axis.imshow(stage_image)
            axis.set_title(name)
            axis.axis("off")
        fig.suptitle(f"{record.image_path.name} | final epoch={epochs} | lesion={record.lesion_risk}, edema={record.edema_risk}")
        fig.tight_layout()
        fig.savefig(output_dir / f"parser_sample_{sample_index:02d}.png", dpi=150)
        plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save six Messidor parser stage visualizations")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="output/parser_samples")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    save_parser_samples(Path(args.data_root), Path(args.output_dir), args.image_size, args.sample_count, args.seed,
                        args.epochs)


if __name__ == "__main__":
    main()
