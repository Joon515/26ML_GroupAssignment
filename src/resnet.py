from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models

try:
    from .preprocess import ALL_PREPROCESS_STEPS, PreprocessConfig, config_from_enabled, load_rgb, preprocess_image
except ImportError:
    from preprocess import ALL_PREPROCESS_STEPS, PreprocessConfig, config_from_enabled, load_rgb, preprocess_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "messidor"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"

@dataclass(frozen=True)
class MessidorRecord:
    image_path: Path
    lesion_risk: int
    edema_risk: int


class MessidorDataset(Dataset[tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]]):
    def __init__(self, records: list[MessidorRecord], preprocess_config: PreprocessConfig, train: bool,
                 seed: int = 0) -> None:
        self.records = records
        self.preprocess_config = preprocess_config
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
        preprocessed = preprocess_image(image, self.preprocess_config, train=self.train, rng=rng)
        tensor = torch.from_numpy(preprocessed).permute(2, 0, 1).float()
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
    records = _read_csv(data_root / "train.csv", data_root / "train")
    records.extend(_read_csv(data_root / "test.csv", data_root / "test"))
    return records


def make_model(arch: str, pretrained: bool = False, classifier: str = "normed") -> nn.Module:
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
    model.fc = MessidorHead(in_features, classifier=classifier)
    return model


class NormedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.weight.data.uniform_(-1.0, 1.0).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))


class MessidorHead(nn.Module):
    def __init__(self, in_features: int, classifier: str = "normed") -> None:
        super().__init__()
        head = NormedLinear if classifier == "normed" else nn.Linear
        self.lesion = head(in_features, 4)
        self.edema = head(in_features, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lesion(x), self.edema(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratification_labels(records: list[MessidorRecord]) -> list[str]:
    return [f"{record.lesion_risk}-{record.edema_risk}" for record in records]


def class_counts(records: list[MessidorRecord], target: str, num_classes: int) -> list[int]:
    counts = [0] * num_classes
    for record in records:
        label = record.lesion_risk if target == "lesion" else record.edema_risk
        counts[label] += 1
    return counts


def _effective_class_weights(counts: list[int], beta: float) -> np.ndarray:
    count_array = np.asarray(counts, dtype=np.float64)
    present = count_array > 0
    weights = np.zeros_like(count_array, dtype=np.float64)
    if not present.any():
        return weights
    if beta <= 0.0:
        weights[present] = 1.0
    else:
        effective_num = 1.0 - np.power(beta, count_array[present])
        weights[present] = (1.0 - beta) / effective_num
    weights[present] = weights[present] / weights[present].sum() * float(present.sum())
    return weights


def drw_weights(counts: list[int], epoch: int, start_epoch: int, beta: float, device: torch.device) -> torch.Tensor | None:
    if start_epoch <= 0 or epoch < start_epoch:
        return None
    weights = _effective_class_weights(counts, beta)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def sample_weights(records: list[MessidorRecord], beta: float = 0.9999,
                   power: float = 0.5, joint_weight: float = 0.5) -> torch.DoubleTensor:
    if power <= 0.0:
        return torch.ones(len(records), dtype=torch.double)
    lesion_weights = _effective_class_weights(class_counts(records, "lesion", 4), beta)
    edema_weights = _effective_class_weights(class_counts(records, "edema", 3), beta)
    joint_counts: dict[str, int] = {}
    for label in stratification_labels(records):
        joint_counts[label] = joint_counts.get(label, 0) + 1
    joint_raw = _effective_class_weights(list(joint_counts.values()), beta)
    joint_weights = dict(zip(joint_counts.keys(), joint_raw, strict=True))

    weights = [
        (lesion_weights[record.lesion_risk]
         + edema_weights[record.edema_risk]
         + joint_weight * joint_weights[f"{record.lesion_risk}-{record.edema_risk}"])
        / (2.0 + joint_weight)
        for record in records
    ]
    weights_array = np.power(np.asarray(weights, dtype=np.float64), power)
    weights_array = weights_array / weights_array.mean()
    return torch.as_tensor(weights_array, dtype=torch.double)


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list: list[int], max_m: float = 0.5, s: float = 30.0) -> None:
        super().__init__()
        safe_counts = np.maximum(np.asarray(cls_num_list, dtype=np.float64), 1.0)
        margins = 1.0 / np.sqrt(np.sqrt(safe_counts))
        margins = margins * (max_m / margins.max())
        self.register_buffer("m_list", torch.as_tensor(margins, dtype=torch.float32))
        self.s = s

    def forward(self, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, target.view(-1, 1), True)
        batch_m = self.m_list[target].view(-1, 1).to(dtype=logits.dtype)
        adjusted_logits = torch.where(index, logits - batch_m, logits)
        return F.cross_entropy(self.s * adjusted_logits, target, weight=weight)


def task_loss(loss_type: str, criterion: LDAMLoss | None, logits: torch.Tensor,
              target: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    if loss_type == "ce":
        return F.cross_entropy(logits, target, weight=weight)
    if criterion is None:
        raise ValueError("LDAM criterion is required when loss_type='ldam'")
    return criterion(logits, target, weight)


def task_metrics(name: str, true: list[int], pred: list[int], labels: list[int]) -> dict[str, object]:
    return {
        f"{name}_accuracy": accuracy_score(true, pred),
        f"{name}_balanced_accuracy": balanced_accuracy_score(true, pred),
        f"{name}_macro_f1": f1_score(true, pred, labels=labels, average="macro", zero_division=0),
        f"{name}_per_class_recall": recall_score(true, pred, labels=labels, average=None, zero_division=0).tolist(),
        f"{name}_confusion_matrix": confusion_matrix(true, pred, labels=labels).tolist(),
    }


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
    metrics = {
        **task_metrics("lesion", lesion_true, lesion_pred, [0, 1, 2, 3]),
        **task_metrics("edema", edema_true, edema_pred, [0, 1, 2]),
    }
    metrics["mean_macro_f1"] = (float(metrics["lesion_macro_f1"]) + float(metrics["edema_macro_f1"])) / 2.0
    metrics["mean_balanced_accuracy"] = (
        float(metrics["lesion_balanced_accuracy"]) + float(metrics["edema_balanced_accuracy"])
    ) / 2.0
    return metrics


def train_one_fold(arch: str, fold: int, train_records: list[MessidorRecord], val_records: list[MessidorRecord],
                   preprocess_config: PreprocessConfig, args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    train_dataset = MessidorDataset(train_records, preprocess_config, train=True, seed=args.seed + fold * 10_000)
    sampler = None
    if args.sampler_power > 0.0:
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(args.seed + fold * 100_003)
        sampler = WeightedRandomSampler(
            sample_weights(train_records, beta=args.sampler_beta, power=args.sampler_power,
                           joint_weight=args.sampler_joint_weight),
            num_samples=len(train_records),
            replacement=True,
            generator=sampler_generator,
        )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(MessidorDataset(val_records, preprocess_config, train=False), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = make_model(arch, pretrained=args.pretrained, classifier=args.classifier).to(device)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lesion_counts = class_counts(train_records, "lesion", 4)
    edema_counts = class_counts(train_records, "edema", 3)
    lesion_criterion = LDAMLoss(lesion_counts, max_m=args.ldam_max_m, s=args.ldam_scale).to(device)
    edema_criterion = LDAMLoss(edema_counts, max_m=args.ldam_max_m, s=args.ldam_scale).to(device)

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        lesion_weights = drw_weights(lesion_counts, epoch, args.drw_start_epoch, args.drw_beta, device)
        edema_weights = drw_weights(edema_counts, epoch, args.drw_start_epoch, args.drw_beta, device)
        model.train()
        running_loss = 0.0
        for images, (lesion_targets, edema_targets) in train_loader:
            images = images.to(device)
            lesion_targets = lesion_targets.to(device)
            edema_targets = edema_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            lesion_logits, edema_logits = model(images)
            loss = (
                task_loss(args.loss_type, lesion_criterion, lesion_logits, lesion_targets, lesion_weights)
                + task_loss(args.loss_type, edema_criterion, edema_logits, edema_targets, edema_weights)
            )
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * images.size(0)
        metrics = evaluate(model, val_loader, device)
        print(json.dumps({"arch": arch, "fold": fold, "epoch": epoch,
                          "loss_type": args.loss_type, "drw_active": lesion_weights is not None,
                          "sampler_active": sampler is not None,
                          "train_loss": running_loss / len(train_records), **metrics}, ensure_ascii=False))

    final_metrics = evaluate(model, val_loader, device)
    final_metrics.update({"arch": arch, "fold": fold, "train_count": len(train_records), "val_count": len(val_records)})
    return final_metrics


def aggregate(results: list[dict[str, object]]) -> dict[str, object]:
    numeric_keys = [
        "lesion_accuracy", "lesion_balanced_accuracy", "lesion_macro_f1",
        "edema_accuracy", "edema_balanced_accuracy", "edema_macro_f1",
        "mean_macro_f1", "mean_balanced_accuracy",
    ]
    summary: dict[str, object] = {"folds": results}
    for key in numeric_keys:
        values = np.array([float(result[key]) for result in results], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=0))
    return summary


def run_cross_validation(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    records = load_messidor_records(Path(args.data_root).expanduser())
    if args.limit:
        records = records[:args.limit]
    preprocess_config = config_from_enabled(args.preprocess_steps, args.image_size)
    labels = stratification_labels(records)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, object] = {
        "preprocess_config": asdict(preprocess_config),
        "training_config": {
            "loss_type": args.loss_type,
            "classifier": args.classifier,
            "optimizer": args.optimizer,
            "pretrained": args.pretrained,
            "drw_start_epoch": args.drw_start_epoch,
            "sampler_power": args.sampler_power,
            "sampler_joint_weight": args.sampler_joint_weight,
        },
        "architectures": {},
    }
    indices = np.arange(len(records))
    for arch in args.architectures:
        arch_results: list[dict[str, object]] = []
        for fold, (train_idx, val_idx) in enumerate(splitter.split(indices, labels), start=1):
            train_records = [records[int(i)] for i in train_idx]
            val_records = [records[int(i)] for i in val_idx]
            arch_results.append(train_one_fold(arch, fold, train_records, val_records, preprocess_config, args, device))
        all_results["architectures"][f"resnet{arch}"] = aggregate(arch_results)  # type: ignore[index]
    architectures = all_results["architectures"]
    best_arch = max(architectures, key=lambda name: architectures[name]["mean_macro_f1_mean"])  # type: ignore[index]
    all_results["best_architecture"] = {
        "name": best_arch,
        "mean_macro_f1": architectures[best_arch]["mean_macro_f1_mean"],  # type: ignore[index]
    }

    result_path = output_dir / "cross_validation_results.json"
    result_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    return all_results


def build_cli() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="Messidor ResNet 5-fold cross validation")
    cli.add_argument("--data-root", dest="data_root", default=str(DEFAULT_DATA_ROOT))
    cli.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "resnet"))
    cli.add_argument("--architectures", nargs="+", default=["18", "34", "50", "101"], choices=["18", "34", "50", "101"])
    cli.add_argument("--preprocess-steps", nargs="+", default=list(ALL_PREPROCESS_STEPS), choices=list(ALL_PREPROCESS_STEPS))
    cli.add_argument("--folds", type=int, default=5)
    cli.add_argument("--epochs", type=int, default=50)
    cli.add_argument("--batch-size", type=int, default=16)
    cli.add_argument("--lr", type=float, default=1e-3)
    cli.add_argument("--weight-decay", type=float, default=1e-4)
    cli.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    cli.add_argument("--momentum", type=float, default=0.9)
    cli.add_argument("--loss-type", default="ldam", choices=["ce", "ldam"])
    cli.add_argument("--classifier", default="normed", choices=["linear", "normed"])
    cli.add_argument("--ldam-max-m", type=float, default=0.5)
    cli.add_argument("--ldam-scale", type=float, default=30.0)
    cli.add_argument("--drw-beta", type=float, default=0.9999)
    cli.add_argument("--drw-start-epoch", type=int, default=31,
                     help="1-based epoch to enable DRW; 0 disables DRW")
    cli.add_argument("--sampler-beta", type=float, default=0.9999)
    cli.add_argument("--sampler-power", type=float, default=0.25,
                     help="0 disables sampler; 1.0 is full effective-number weighting")
    cli.add_argument("--sampler-joint-weight", type=float, default=0.25,
                     help="Contribution of the joint lesion-edema label to sample weights")
    cli.add_argument("--image-size", type=int, default=224)
    cli.add_argument("--num-workers", type=int, default=2)
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--device", default="")
    cli.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    cli.add_argument("--limit", type=int, default=0, help="Smoke-test limit; leave 0 for full dataset")
    return cli


def main() -> None:
    args = build_cli().parse_args()
    run_cross_validation(args)


if __name__ == "__main__":
    main()
