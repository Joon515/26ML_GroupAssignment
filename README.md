# Messidor 视网膜风险分类实验

本项目用于 Messidor 眼底图像数据集的多任务分类实验：同时预测 `lesion risk`（4 类）和 `Risk of macular edema`（3 类）。代码包含数据准备、预处理、ResNet 训练、交叉验证、预处理消融和预处理可视化。

## 当前实现概览

- 数据集：Messidor/Kaggle，`setup.sh` 会下载并整理到 `data/messidor/`。
- 模型：ResNet18、ResNet34、ResNet50、ResNet101。
- 任务：一个共享 ResNet backbone，两个分类头：
  - `lesion`：4 类。
  - `edema`：3 类。
- 训练方式：Stratified 5-fold cross validation。
- 类别不平衡：训练集使用改进的 `WeightedRandomSampler`，loss 使用 LDAM-DRW。
- 预处理：resize、可选随机旋转、可选 CLAHE、可选 ImageNet 归一化。
- 运行设备：默认自动使用 CUDA；如果 `torch.cuda.is_available()` 为 false，则退回 CPU。

## 目录结构

```text
.
├── main.py                    # 默认入口；等价于 python -m src.resnet
├── setup.sh                   # 下载数据、移动 baseline model、创建 conda 环境
├── environment.yml            # conda 环境配置，环境名 26ml
├── src/
│   ├── preprocess.py          # 图像读取、预处理、预处理阶段可视化逻辑
│   ├── preprocess_demo.py     # 导出预处理过程示例图
│   ├── resnet.py              # 数据读取、模型、训练、评估、交叉验证
│   ├── run_ablation.py        # 预处理消融实验入口
│   └── visualize_results.py   # 根据训练 JSON 生成手写风格结果图
├── data/messidor/             # setup.sh 生成；包含 train/test 图片和 CSV
├── baseline/                  # setup.sh 移出的 mpiotte-standard.model
└── output/                    # 训练结果和可视化输出
```

## 环境与数据准备

首次运行：

```bash
./setup.sh
```

`setup.sh` 做三件事：

1. 检查 `unzip`、`aria2c`、`conda`。
2. 从 Kaggle URL 下载 Messidor 数据集并解压到 `data/messidor/`。
3. 如果 conda 环境 `26ml` 不存在，则根据 `environment.yml` 创建。

之后激活环境：

```bash
conda activate 26ml
```

也可以不激活，直接使用：

```bash
conda run -n 26ml python main.py
```

## 预处理实现

实现文件：`src/preprocess.py`

核心配置：`PreprocessConfig`

```python
@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int = 224
    normalize: bool = True
    random_rotation: bool = True
    clahe: bool = True
    rotation_degrees: float = 20.0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
```

当前保留的预处理步骤：

1. `resize`：所有图片 resize 成 `image_size x image_size`。
2. `rotate`：训练阶段启用；在 `[-20°, 20°]` 内随机旋转，验证阶段不启用。
3. `clahe`：在 LAB 色彩空间对 L 通道做 CLAHE，再转回 RGB。
4. `normalize`：使用 ImageNet mean/std：
   - mean = `[0.485, 0.456, 0.406]`
   - std = `[0.229, 0.224, 0.225]`

已明确移除：

- 去黑边 / crop black border。
- color jitter。

可消融的步骤只剩：

```python
ALL_PREPROCESS_STEPS = ("normalize", "rotate", "clahe")
```

## 模型与训练实现

实现文件：`src/resnet.py`

### 数据读取

`load_messidor_records(data_root)` 会读取：

- `data/messidor/train.csv` + `data/messidor/train/`
- `data/messidor/test.csv` + `data/messidor/test/`

然后合并成一个记录列表，再做交叉验证切分。

每条记录包含：

```python
@dataclass(frozen=True)
class MessidorRecord:
    image_path: Path
    lesion_risk: int
    edema_risk: int
```

### Dataset

`MessidorDataset.__getitem__()` 流程：

1. 读取 RGB 图片。
2. 按当前 `PreprocessConfig` 做预处理。
3. 转成 PyTorch tensor，形状从 `HWC` 变成 `CHW`。
4. 返回：

```python
image_tensor, (lesion_label, edema_label)
```

训练集会根据 `epoch`、`seed`、`index` 生成确定性的随机数，用于随机旋转；验证集不做随机旋转。

### 模型结构

`make_model()` 支持：

- `18` → ResNet18
- `34` → ResNet34
- `50` → ResNet50
- `101` → ResNet101

最后的 `fc` 被替换为双头分类器：

```python
class MessidorHead(nn.Module):
    self.lesion = nn.Linear(in_features, 4)
    self.edema = nn.Linear(in_features, 3)
```

forward 输出：

```python
lesion_logits, edema_logits
```

### Loss

两个任务默认分别使用 LDAM loss，然后相加：

```python
loss = LDAM(lesion_logits, lesion_targets, lesion_drw_weights) \
     + LDAM(edema_logits, edema_targets, edema_drw_weights)
```

可用 `--loss-type ce` 切回交叉熵。LDAM margin 按每个任务的类别样本数计算；分类头默认使用 LDAM-DRW 论文中的 normalized linear head（`--classifier normed`）。

DRW 默认从第 31 个 epoch 启用 effective-number class weights；`--drw-start-epoch 0` 可关闭 DRW。

### 类别不平衡处理

训练集默认使用轻量 `WeightedRandomSampler`。采样权重不再只看组合标签，而是融合：

```python
lesion effective-number weight
edema effective-number weight
joint lesion-edema effective-number weight
```

默认 `--sampler-power 0.25`、`--sampler-joint-weight 0.25`，避免 sampler + DRW 对少数类双重过补偿；`--sampler-power 0` 可完全关闭 sampler。

### 交叉验证

`run_cross_validation()` 使用：

```python
StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
```

分层标签同样是：

```python
f"{lesion_risk}-{edema_risk}"
```

默认 `folds=5`。

### 评估指标

每个 fold 结束后输出：

- lesion / edema accuracy
- lesion / edema balanced accuracy
- lesion / edema macro-F1
- lesion / edema per-class recall
- lesion / edema confusion matrix
- `mean_macro_f1`：两个任务 macro-F1 的平均值，默认用于比较模型

每个架构最终会聚合 mean/std，并写入：

```text
output/resnet/cross_validation_results.json
```

## 运行方式

### 1. 默认训练：比较不同 ResNet 架构

```bash
conda run -n 26ml python main.py
```

`main.py` 等价于：

```bash
conda run -n 26ml python -m src.resnet
```

默认会跑：

- ResNet18
- ResNet34
- ResNet50
- ResNet101

默认预处理为：

```text
normalize + rotate + clahe
```

注意：`main.py` 不会自动做预处理消融；它只在同一套预处理下比较 ResNet 架构。

### 2. 指定模型、epoch、batch size

```bash
conda run -n 26ml python main.py \
  --architectures 18 34 \
  --epochs 50 \
  --batch-size 16
```

关键训练参数默认值：

```bash
--pretrained \
--optimizer sgd \
--momentum 0.9 \
--lr 0.001 \
--loss-type ldam \
--classifier normed \
--ldam-max-m 0.5 \
--ldam-scale 30 \
--drw-beta 0.9999 \
--drw-start-epoch 31 \
--sampler-beta 0.9999 \
--sampler-power 0.25 \
--sampler-joint-weight 0.25
```

### 3. 指定预处理步骤

```bash
conda run -n 26ml python main.py \
  --preprocess-steps normalize clahe
```

可选步骤只有：

```text
normalize rotate clahe
```

### 4. 显式指定 CUDA 或 CPU

默认自动选择 CUDA：

```python
"cuda" if torch.cuda.is_available() else "cpu"
```

也可以手动指定：

```bash
conda run -n 26ml python main.py --device cuda
conda run -n 26ml python main.py --device cpu
```

### 5. 快速 smoke test

只取前 N 条数据：

```bash
conda run -n 26ml python main.py \
  --architectures 18 \
  --folds 2 \
  --epochs 1 \
  --limit 64 \
  --image-size 64 \
  --batch-size 8
```

## 预处理消融实验

入口：`src/run_ablation.py`

运行：

```bash
conda run -n 26ml python -m src.run_ablation
```

预处理消融组合：

| 名称 | 步骤 |
| --- | --- |
| `none` | 无额外预处理，只 resize 并缩放到 `[0, 1]` |
| `normalize` | normalize |
| `clahe` | CLAHE |
| `rotate` | random rotation |
| `normalize_clahe` | normalize + CLAHE |
| `normalize_rotate` | normalize + random rotation |
| `clahe_rotate` | CLAHE + random rotation |
| `full` | normalize + random rotation + CLAHE |

不平衡策略消融可用：

```bash
conda run -n 26ml python -m src.run_ablation --ablation-kind imbalance
```

包含：

```text
ce_no_sampler
ce_sampler
ldam_only
ldam_drw
ldam_drw_sampler
```

也可以用 `--ablation-kind both` 同时运行预处理和不平衡策略消融。每个消融组合都会调用一次 `run_cross_validation()`，因此默认情况下每个组合都会比较 ResNet18/34/50/101。

可以减少模型或 epoch 来缩短时间：

```bash
conda run -n 26ml python -m src.run_ablation \
  --architectures 18 \
  --epochs 1 \
  --limit 64 \
  --image-size 64
```

## 导出预处理过程图

入口：`src/preprocess_demo.py`

运行：

```bash
conda run -n 26ml python -m src.preprocess_demo
```

默认导出 6 张样本图到：

```text
output/preprocess_samples/
```

每张图展示：

```text
original -> resize -> random_rotation -> clahe -> normalize
```

如果某个步骤在配置中关闭，则不会出现在图中。

这个脚本只用于展示预处理示意效果，不训练模型，也不关联训练 epoch。

## 输出 JSON 结构

训练输出文件：

```text
cross_validation_results.json
```

顶层结构：

```json
{
  "preprocess_config": {
    "image_size": 224,
    "normalize": true,
    "random_rotation": true,
    "clahe": true,
    "rotation_degrees": 20.0,
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid_size": 8
  },
  "architectures": {
    "resnet18": {
      "folds": [],
      "lesion_accuracy_mean": 0.0,
      "lesion_accuracy_std": 0.0,
      "lesion_macro_f1_mean": 0.0,
      "lesion_macro_f1_std": 0.0,
      "edema_accuracy_mean": 0.0,
      "edema_accuracy_std": 0.0,
      "edema_macro_f1_mean": 0.0,
      "edema_macro_f1_std": 0.0
    }
  }
}
```

## 可视化训练结果

入口：`src/visualize_results.py`

这个脚本不训练模型，只读取训练完成后的 `cross_validation_results.json`，再生成手写风格的 PNG 图。

默认读取：

```bash
conda run -n 26ml python -m src.visualize_results
```

默认输入是整个 ablation 输出目录，因此不传参就会可视化全部消融结果：

```text
output/ablations/
```

默认输出到：

```text
output/figures/
```

也可以指定单个结果文件：

```bash
conda run -n 26ml python -m src.visualize_results \
  output/ablations/full/cross_validation_results.json \
  --output-dir output/figures
```

或指定一个目录，脚本会递归寻找其中所有 `cross_validation_results.json`：

```bash
conda run -n 26ml python -m src.visualize_results \
  output/ablations \
  --output-dir output/figures
```

每个结果文件会生成三类图：

- `*_metric_summary.png`：不同 ResNet 架构的 accuracy / macro-F1 柱状对比，误差线表示各 fold 的最小值到最大值。
- `*_fold_ranges.png`：只展示每个架构在各 fold 上的均值、下限和上限，不再画逐 fold 折线。
- `*_confusion_matrices.png`：按 fold 汇总后的 lesion / edema 混淆矩阵。

当输入目录下有多个结果文件时，还会额外生成：

- `ablation_overview.png`：不同预处理消融组合之间的整体对比，纵轴自动缩放到有效区间以突出差异。
- `ablation_delta_overview.png`：以 `none` 为基线，直接展示各消融组合带来的指标增减。

## 常见注意点

- `python main.py`：做 ResNet 架构对比，不做预处理消融。
- `python -m src.run_ablation`：做预处理消融，并且默认每个消融组合都比较多个 ResNet 架构。
- 训练默认会自动使用 CUDA；如果显存不够，减小 `--batch-size`、`--image-size` 或只跑部分 `--architectures`。
- 当前预处理没有去黑边，也没有 color jitter。
- `--limit` 只用于快速测试，不应用于最终实验结果。
