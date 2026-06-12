from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from typing import Any, cast
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models

try:
    from .parser import ALL_PARSER_STEPS, ParserConfig, config_from_enabled, load_rgb, parse_image
except ImportError:
    from parser import ALL_PARSER_STEPS, ParserConfig, config_from_enabled, load_rgb, parse_image


@dataclass(frozen=True)
class MessidorRecord:
    image_path: Path
    lesion_risk: int
    edema_risk: int


class MessidorDataset(Dataset[tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]]):
    def __init__(self, records: list[MessidorRecord], parser_config: ParserConfig, train: bool,
                 seed: int = 0) -> None:
        self.records = records
        self.parser_config = parser_config
        self.train = train
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        record = self.records[index]
        image = load_rgb(record.image_path)
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index) if self.train else None
        parsed = parse_image(image, self.parser_config, train=self.train, rng=rng)
        tensor = torch.from_numpy(parsed).permute(2, 0, 1).float()
        return tensor, (torch.tensor(record.lesion_risk, dtype=torch.long),
                        torch.tensor(record.edema_risk, dtype=torch.long))


def _read_csv(csv_path: Path, image_dir: Path) -> list[MessidorRecord]:
    records: list[MessidorRecord] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_name = row["Image"].strip()
            lesion = int(row["Id"].strip())
            edema_key = next(key for key in row if key.strip() == "Risk of macular edema")
            edema = int(row[edema_key].strip())
            image_path = image_dir / image_name
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            if not 0 <= lesion <= 3:
                raise ValueError(f"Invalid lesion risk {lesion} in {csv_path}")
            if not 0 <= edema <= 2:
                raise ValueError(f"Invalid edema risk {edema} in {csv_path}")
            records.append(MessidorRecord(image_path, lesion, edema))
    return records


def load_messidor_records(data_root: Path) -> list[MessidorRecord]:
    dataset = data_root / "dataset"
    records = _read_csv(dataset / "train.csv", dataset / "train")
    records.extend(_read_csv(dataset / "test.csv", dataset / "test"))
    return records


def make_model(arch: str, pretrained: bool = False) -> nn.Module:
    builders = {
        "18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
        "34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
        "50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        "101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
    }
    if arch not in builders:
        raise ValueError(f"Unsupported ResNet architecture: {arch}")
    builder, weights = builders[arch]
    model = builder(weights=weights if pretrained else None)
    in_features = model.fc.in_features
    model.fc = MessidorHead(in_features)
    return model


class MessidorHead(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.lesion = nn.Linear(in_features, 4)
        self.edema = nn.Linear(in_features, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lesion(x), self.edema(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratification_labels(records: list[MessidorRecord]) -> list[str]:
    return [f"{record.lesion_risk}-{record.edema_risk}" for record in records]
def sample_weights(records: list[MessidorRecord]) -> torch.DoubleTensor:
    counts: dict[str, int] = {}
    for label in stratification_labels(records):
        counts[label] = counts.get(label, 0) + 1
    weights = [1.0 / counts[f"{record.lesion_risk}-{record.edema_risk}"] for record in records]
    return torch.DoubleTensor(weights)




def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    lesion_true: list[int] = []
    lesion_pred: list[int] = []
    edema_true: list[int] = []
    edema_pred: list[int] = []
    with torch.no_grad():
        for images, (lesion_targets, edema_targets) in loader:
            images = images.to(device)
            lesion_logits, edema_logits = model(images)
            lesion_true.extend(lesion_targets.tolist())
            edema_true.extend(edema_targets.tolist())
            lesion_pred.extend(lesion_logits.argmax(dim=1).cpu().tolist())
            edema_pred.extend(edema_logits.argmax(dim=1).cpu().tolist())
    return {
        "lesion_accuracy": accuracy_score(lesion_true, lesion_pred),
        "lesion_macro_f1": f1_score(lesion_true, lesion_pred, average="macro", zero_division=0),
        "lesion_confusion_matrix": confusion_matrix(lesion_true, lesion_pred, labels=[0, 1, 2, 3]).tolist(),
        "edema_accuracy": accuracy_score(edema_true, edema_pred),
        "edema_macro_f1": f1_score(edema_true, edema_pred, average="macro", zero_division=0),
        "edema_confusion_matrix": confusion_matrix(edema_true, edema_pred, labels=[0, 1, 2]).tolist(),
    }


def train_one_fold(arch: str, fold: int, train_records: list[MessidorRecord], val_records: list[MessidorRecord],
                   parser_config: ParserConfig, args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    train_dataset = MessidorDataset(train_records, parser_config, train=True, seed=args.seed + fold * 10_000)
    sampler = WeightedRandomSampler(sample_weights(train_records), num_samples=len(train_records), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(MessidorDataset(val_records, parser_config, train=False), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = make_model(arch, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        for images, (lesion_targets, edema_targets) in train_loader:
            images = images.to(device)
            lesion_targets = lesion_targets.to(device)
            edema_targets = edema_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            lesion_logits, edema_logits = model(images)
            loss = criterion(lesion_logits, lesion_targets) + criterion(edema_logits, edema_targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * images.size(0)
        metrics = evaluate(model, val_loader, device)
        print(json.dumps({"arch": arch, "fold": fold, "epoch": epoch,
                          "train_loss": running_loss / len(train_records), **metrics}, ensure_ascii=False))

    final_metrics = evaluate(model, val_loader, device)
    final_metrics.update({"arch": arch, "fold": fold, "train_count": len(train_records), "val_count": len(val_records)})
    return final_metrics


def aggregate(results: list[dict[str, object]]) -> dict[str, object]:
    numeric_keys = ["lesion_accuracy", "lesion_macro_f1", "edema_accuracy", "edema_macro_f1"]
    summary: dict[str, object] = {"folds": results}
    for key in numeric_keys:
        values = np.array([float(result[key]) for result in results], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=0))
    return summary


def run_cross_validation(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    records = load_messidor_records(Path(args.data_root))
    if args.limit:
        records = records[:args.limit]
    parser_config = config_from_enabled(args.parser_steps, args.image_size)
    labels = stratification_labels(records)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, object] = {"parser_config": asdict(parser_config), "architectures": {}}
    indices = np.arange(len(records))
    for arch in args.architectures:
        arch_results: list[dict[str, object]] = []
        for fold, (train_idx, val_idx) in enumerate(splitter.split(indices, labels), start=1):
            train_records = [records[int(i)] for i in train_idx]
            val_records = [records[int(i)] for i in val_idx]
            arch_results.append(train_one_fold(arch, fold, train_records, val_records, parser_config, args, device))
        all_results["architectures"][f"resnet{arch}"] = aggregate(arch_results)  # type: ignore[index]

    result_path = output_dir / "cross_validation_results.json"
    result_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    return all_results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Messidor ResNet 5-fold cross validation")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="output/resnet")
    parser.add_argument("--architectures", nargs="+", default=["18", "34", "50", "101"], choices=["18", "34", "50", "101"])
    parser.add_argument("--parser-steps", nargs="+", default=list(ALL_PARSER_STEPS), choices=list(ALL_PARSER_STEPS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test limit; leave 0 for full dataset")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_cross_validation(args)


if __name__ == "__main__":
    main()
