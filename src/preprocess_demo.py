from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


def handwritten_font() -> str:
    for family in ("Comic Sans MS", "Comic Neue", "DejaVu Sans"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except ValueError:
            pass
    return "DejaVu Sans"


def apply_handwritten_style() -> None:
    plt.rcParams.update({
        "font.family": handwritten_font(),
        "path.sketch": (0.6, 80.0, 1.0),
        "font.weight": "regular",
        "font.size": 10,
        "axes.titlesize": 11,
        "figure.facecolor": "#fffdf5",
        "axes.facecolor": "#fffdf5",
        "text.color": "#222222",
        "savefig.facecolor": "#fffdf5",
    })

try:
    from .preprocess import PreprocessConfig, load_rgb, preprocess_stages
    from .resnet import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, load_messidor_records
except ImportError:
    from preprocess import PreprocessConfig, load_rgb, preprocess_stages
    from resnet import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, load_messidor_records


def save_preprocess_samples(data_root: Path, output_dir: Path, image_size: int, sample_count: int, seed: int) -> None:
    records = load_messidor_records(data_root)
    rng = random.Random(seed)
    selected = rng.sample(records, k=min(sample_count, len(records)))
    apply_handwritten_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_sample in output_dir.glob("preprocess_sample_*.png"):
        old_sample.unlink()

    config = PreprocessConfig(image_size=image_size)
    for sample_index, record in enumerate(selected, start=1):
        image = load_rgb(record.image_path)
        stages = preprocess_stages(image, config, rng=random.Random(seed + sample_index))
        fig, axes = plt.subplots(1, len(stages), figsize=(3 * len(stages), 3))
        if len(stages) == 1:
            axes = [axes]
        for axis, (name, stage_image) in zip(axes, stages):
            axis.imshow(stage_image)
            axis.set_title(name)
            axis.axis("off")
        fig.suptitle(f"{record.image_path.name} | lesion={record.lesion_risk}, edema={record.edema_risk}")
        fig.tight_layout()
        fig.savefig(output_dir / f"preprocess_sample_{sample_index:02d}.png", dpi=150)
        plt.close(fig)


def build_cli() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="Save six Messidor preprocess stage visualizations")
    cli.add_argument("--data-root", dest="data_root", default=str(DEFAULT_DATA_ROOT))
    cli.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "preprocess_samples"))
    cli.add_argument("--image-size", type=int, default=224)
    cli.add_argument("--sample-count", type=int, default=6)
    cli.add_argument("--seed", type=int, default=42)
    return cli


def main() -> None:
    args = build_cli().parse_args()
    save_preprocess_samples(Path(args.data_root).expanduser(), Path(args.output_dir).expanduser(), args.image_size, args.sample_count, args.seed)


if __name__ == "__main__":
    main()
