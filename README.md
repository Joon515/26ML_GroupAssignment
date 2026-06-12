# 简陋的README
## 注意事项
- 数据集来自[eyepacs-aptos-messidor-diabetic-retinopathy/kaggle](https://www.kaggle.com/api/v1/datasets/download/ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy)和[Meddidor/kaggle](https://www.kaggle.com/api/v1/datasets/download/hanhan2010/messidor)，至于为什么不只用Messidor，是在首次训练resnet时发现了严重的数据不平衡问题，打算用WeightedRandomSampler解决，但发现样本太少。
- parser用于对数据的预处理，包括去黑边、resize、rotation、color_jitter、clahe、归一化，demo用于导出最后一epoch的图片，以便写报告。
- 感觉可以很适合做关于预处理的消融实验。