# 简陋的README

## 注意事项

- 数据集来自 [Messidor/Kaggle](https://www.kaggle.com/api/v1/datasets/download/hanhan2010/messidor)，执行 `./setup.sh` 后会放在 `data/messidor/`，基线模型会放在 `baseline/mpiotte-standard.model`。
- preprocess 用于对数据的预处理，包括 resize、rotation、clahe、归一化，demo 用于导出最后一 epoch 的图片，以便写报告。
- ResNet 训练默认读取 `data/messidor/`，输出默认写入 `output/resnet/`。

