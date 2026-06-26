# Messidor 视网膜风险多任务分类实验报告

## 1. 实验目标

本实验针对 Messidor 眼底图像数据集进行多任务风险分类，同时预测：

- `lesion risk`：4 类，标签为 0/1/2/3。
- `Risk of macular edema`：3 类，标签为 0/1/2。

实验重点是处理 Messidor 数据集类别不平衡问题，并比较不同 ResNet 架构、预处理方案和不平衡学习策略对分类性能的影响。

## 2. 数据集与类别分布

本次实验共使用 1200 张图像。类别分布如下。

### 2.1 Lesion risk 分布

| 类别 | 样本数 | 占比 |
|---:|---:|---:|
| 0 | 546 | 45.50% |
| 1 | 153 | 12.75% |
| 2 | 247 | 20.58% |
| 3 | 254 | 21.17% |

### 2.2 Macular edema risk 分布

| 类别 | 样本数 | 占比 |
|---:|---:|---:|
| 0 | 974 | 81.17% |
| 1 | 75 | 6.25% |
| 2 | 151 | 12.58% |

### 2.3 联合标签分布

交叉验证分层标签使用：

```python
f"{lesion_risk}-{edema_risk}"
```

联合标签分布如下。

| 联合标签 | 样本数 |
|---|---:|
| 0-0 | 546 |
| 2-0 | 182 |
| 1-0 | 142 |
| 3-2 | 108 |
| 3-0 | 104 |
| 3-1 | 42 |
| 2-2 | 37 |
| 2-1 | 28 |
| 1-2 | 6 |
| 1-1 | 5 |

数据不平衡非常明显，尤其是 edema=1 和联合标签 1-1、1-2。因此本实验主要使用 macro-F1、balanced accuracy 和 per-class recall 作为评估依据，而不是只看 accuracy。

## 3. 方法

### 3.1 模型结构

使用 ImageNet 预训练 ResNet 作为共享 backbone，并接两个分类头：

- lesion head：4 类输出。
- edema head：3 类输出。

最佳配置中使用 `resnet101`，分类头采用 normalized linear classifier。该设计更接近 LDAM-DRW 原始范式，有助于长尾分类下的 margin-based learning。

### 3.2 Loss 与不平衡处理

最终主实验采用：

```text
LDAM loss + DRW + mild WeightedRandomSampler
```

具体设置：

| 参数 | 设置 |
|---|---|
| loss | LDAM |
| classifier | normalized linear |
| optimizer | SGD |
| pretrained | true |
| DRW start epoch | 31 |
| sampler power | 0.25 |
| sampler joint weight | 0.25 |

LDAM 根据每个任务的类别样本数设置类别 margin。DRW 在训练后期启用 effective-number class weights。WeightedRandomSampler 不再只依赖联合标签反比采样，而是融合 lesion、edema 和联合标签的 effective-number 权重，并使用较弱的 `sampler_power=0.25` 防止少数类过度重复采样。

### 3.3 交叉验证

使用 Stratified 5-fold cross validation。每个 fold 训练集 960 张，验证集 240 张。分层依据为 lesion-edema 联合标签。

## 4. 主实验：不同 ResNet 架构对比

主实验固定预处理为：

```text
normalize + random rotation + CLAHE
```

固定训练策略为 LDAM-DRW + normalized classifier + pretrained backbone。

| 架构 | Mean Macro-F1 | Lesion Acc | Lesion BAcc | Lesion Macro-F1 | Edema Acc | Edema BAcc | Edema Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 | 0.5197 ± 0.0363 | 0.5050 | 0.4760 | 0.4578 | 0.8350 | 0.5965 | 0.5816 |
| ResNet34 | 0.5087 ± 0.0332 | 0.5442 | 0.4736 | 0.4624 | 0.7733 | 0.6065 | 0.5550 |
| ResNet50 | 0.5598 ± 0.0293 | 0.5758 | 0.4883 | 0.4891 | 0.8542 | 0.6394 | 0.6304 |
| ResNet101 | **0.5658 ± 0.0261** | **0.5825** | **0.4972** | **0.4992** | **0.8592** | **0.6446** | **0.6323** |

结论：ResNet101 的 mean macro-F1 最高，为 0.5658；ResNet50 非常接近，为 0.5598。考虑性能，ResNet101 是最佳模型；考虑计算成本，ResNet50 是更轻量的备选。

## 5. 最佳模型详细结果：ResNet101

### 5.1 5-fold 汇总指标

| 指标 | Mean ± Std |
|---|---:|
| mean macro-F1 | **0.5658 ± 0.0261** |
| mean balanced accuracy | **0.5709 ± 0.0171** |
| lesion accuracy | 0.5825 ± 0.0241 |
| lesion balanced accuracy | 0.4972 ± 0.0122 |
| lesion macro-F1 | 0.4992 ± 0.0144 |
| edema accuracy | 0.8592 ± 0.0252 |
| edema balanced accuracy | 0.6446 ± 0.0369 |
| edema macro-F1 | 0.6323 ± 0.0481 |

### 5.2 每个 fold 的结果

| Fold | Mean F1 | Lesion F1 | Edema F1 | Lesion Acc | Edema Acc |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.5591 | 0.4745 | 0.6437 | 0.5417 | 0.8375 |
| 2 | **0.6078** | **0.5192** | **0.6964** | 0.6000 | **0.8833** |
| 3 | 0.5717 | 0.4994 | 0.6440 | 0.5792 | 0.8750 |
| 4 | 0.5642 | 0.4983 | 0.6301 | **0.6125** | 0.8792 |
| 5 | 0.5262 | 0.5049 | 0.5475 | 0.5792 | 0.8208 |

Fold 2 表现最好，mean macro-F1 达到 0.6078。Fold 5 的 edema macro-F1 较低，是整体方差的主要来源之一。

### 5.3 Lesion 混淆矩阵与逐类表现

聚合 5 个 fold 后的 lesion confusion matrix：

| True \\ Pred | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 407 | 74 | 52 | 13 |
| 1 | 76 | 33 | 38 | 6 |
| 2 | 84 | 39 | 77 | 47 |
| 3 | 33 | 7 | 32 | 182 |

逐类 precision / recall：

| Lesion class | Support | Predicted count | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0 | 546 | 600 | 0.6783 | **0.7454** |
| 1 | 153 | 153 | 0.2157 | 0.2157 |
| 2 | 247 | 199 | 0.3869 | 0.3117 |
| 3 | 254 | 248 | **0.7339** | 0.7165 |

Lesion 任务中，模型对两端类别 0 和 3 识别较好，但对中间类别 1 和 2 表现较弱。这说明模型能区分明显的轻/重风险，但对中间等级边界仍然混淆。

### 5.4 Edema 混淆矩阵与逐类表现

聚合 5 个 fold 后的 edema confusion matrix：

| True \\ Pred | 0 | 1 | 2 |
|---:|---:|---:|---:|
| 0 | 896 | 24 | 54 |
| 1 | 39 | 18 | 18 |
| 2 | 29 | 5 | 117 |

逐类 precision / recall：

| Edema class | Support | Predicted count | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0 | 974 | 964 | **0.9295** | **0.9199** |
| 1 | 75 | 47 | 0.3830 | 0.2400 |
| 2 | 151 | 189 | 0.6190 | 0.7748 |

Edema 任务中，类别 0 和 2 表现较好；类别 1 是主要瓶颈。该类别样本数只有 75，占比 6.25%，且医学上也可能位于边界状态，因此较难学习。

## 6. 预处理消融实验

当前 `output/ablations/ablation_results.json` 中保存的是旧版预处理消融结果，只包含 `none` 与 `full` 两组。由于它不是最新 LDAM-DRW + pretrained 主实验配置下重新跑出的完整 ablation，因此不能与第 4 节主实验绝对值直接比较；但它可以用于观察预处理趋势。

### 6.1 none vs full

`none`：仅 resize 和缩放到 [0, 1]。

`full`：normalize + random rotation + CLAHE。

| Preprocess | Arch | Mean Macro-F1 | Lesion Acc | Lesion F1 | Edema Acc | Edema F1 |
|---|---|---:|---:|---:|---:|---:|
| none | ResNet18 | 0.3222 | 0.3283 | 0.2770 | **0.6508** | 0.3673 |
| full | ResNet18 | **0.3363** | 0.3108 | **0.2843** | 0.5858 | **0.3883** |
| none | ResNet34 | 0.2320 | 0.2500 | 0.2151 | 0.3700 | 0.2489 |
| full | ResNet34 | **0.2780** | **0.2750** | **0.2548** | **0.4017** | **0.3013** |
| none | ResNet50 | **0.2170** | **0.2617** | **0.1961** | **0.3542** | **0.2379** |
| full | ResNet50 | 0.1889 | 0.2375 | 0.1889 | 0.2758 | 0.1889 |
| none | ResNet101 | **0.2477** | **0.2925** | **0.2332** | 0.3958 | 0.2623 |
| full | ResNet101 | 0.2291 | 0.2250 | 0.1790 | **0.4233** | **0.2792** |

趋势：full preprocessing 对 ResNet18 和 ResNet34 的 macro-F1 有提升，但对 ResNet50 和 ResNet101 不稳定。旧实验中 ResNet18 + full 最好。新版主实验表明，在加入 pretrained、normalized head 和更合理的不平衡学习后，更深的 ResNet50/101 才明显受益。

## 7. 与旧最佳结果的对比

旧输出中的最佳配置为：

```text
full preprocessing + ResNet18
mean macro-F1 = 0.3363
```

新版主实验最佳配置为：

```text
full preprocessing + pretrained ResNet101 + LDAM-DRW + normed classifier
mean macro-F1 = 0.5658
```

提升：

```text
0.5658 - 0.3363 = +0.2295
```

这说明性能提升主要来自训练范式改进，而不仅是预处理本身。关键因素包括 ImageNet 预训练、normalized classifier、LDAM-DRW、较长训练和更温和的 sampler。

## 8. 讨论

### 8.1 为什么 accuracy 不能作为唯一指标

Edema 类别 0 占 81.17%。如果模型总是预测 edema=0，也能获得很高 accuracy，但不能有效识别少数类。因此本实验更关注 macro-F1 和 balanced accuracy。

最佳 ResNet101 的 edema accuracy 为 0.8592，高于多数类基线 accuracy 0.8117；同时 edema macro-F1 达到 0.6323，说明模型不仅预测多数类，也学习到了类别 2 的判别特征。不过 edema=1 的 recall 仍只有 0.2400，是后续改进重点。

### 8.2 当前模型的主要错误来源

1. Lesion 中间等级 1/2 混淆较多。
2. Edema=1 样本太少，recall 低。
3. 当前 loss 仍将等级视为普通离散类别，没有显式利用风险等级的有序性。

### 8.3 后续改进方向

1. 引入 ordinal regression 或 classification + ordinal auxiliary loss，利用 0/1/2/3 的等级顺序。
2. 针对 lesion=1/2 和 edema=1 做更细粒度的数据增强或代价敏感学习。
3. 重新运行新版完整 ablation：

```bash
python -m src.run_ablation \
  --ablation-kind both \
  --architectures 50 101
```

4. 在报告最终结果时，同时报告 macro-F1、balanced accuracy、per-class recall 和 confusion matrix，避免 accuracy 掩盖少数类问题。

## 9. 结论

本实验最终最佳模型为 pretrained ResNet101 + normalized classifier + LDAM-DRW + mild WeightedRandomSampler。其 5-fold mean macro-F1 达到 0.5658，明显优于旧实验最佳的 0.3363。模型对 lesion 两端类别和 edema 0/2 类别表现较好，但对 lesion 中间等级和 edema=1 仍存在明显不足。整体来看，针对类别不平衡的训练策略和 ImageNet 预训练是提升 Messidor 分类性能的关键。