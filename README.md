# Rao2016 Image Forgery Detection - Simple PyTorch Reproduction

这是根据 `agent_exp_guide.md` 搭建的简化实验环境，用轻量 CNN 复现 Rao2016 图像篡改检测的核心流程：

```text
image dataset -> 128x128 patch/image transform -> CNN binary classifier -> metrics/plots
```

原论文使用 patch CNN + SRM 高通滤波器初始化 + SVM。本项目保留 patch-based CNN 二分类主线，降低复现难度，适合课程作业展示。

## 1. 环境

当前项目已创建虚拟环境：

```bash
source .venv/bin/activate
```

安装依赖：

```bash
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

`requirements.txt` 默认使用 PyTorch CPU wheel，避免在普通课程实验环境里下载体积很大的 CUDA 运行库。

如果下载 CASIA 等真实篡改数据集不方便，可以先用合成玩具数据验证完整流程。

## 2. 数据目录

真实数据或合成数据都整理成：

```text
dataset/
  train/
    authentic/
    forged/
  val/
    authentic/
    forged/
  test/
    authentic/
    forged/
```

标签约定：

```text
authentic = 0
forged = 1
```

## 3. 生成玩具数据

```bash
python -m forgery_exp.make_toy_dataset --output dataset --size 128 --train 200 --val 50 --test 50
```

玩具数据会生成两类图像：

- `authentic`: 平滑纹理和几何形状组成的“原始图像”
- `forged`: 在原图上复制或拼接一个矩形区域，模拟 copy-move / splicing 边界

## 4. 训练

```bash
python -m forgery_exp.train --data-dir dataset --epochs 20 --batch-size 32 --output-dir runs/rao2016_simple
```

输出文件：

```text
runs/rao2016_simple/
  best_model.pt
  metrics.json
  loss_accuracy.png
  confusion_matrix.png
  sample_predictions.png
```

快速 smoke test 可以把 epoch 调小：

```bash
python -m forgery_exp.train --data-dir dataset --epochs 1 --batch-size 16 --output-dir runs/smoke
```

## 5. 单张图片预测

```bash
python -m forgery_exp.predict --checkpoint runs/rao2016_simple/best_model.pt --image dataset/test/forged/forged_0000.png
```

## 6. 报告可写内容

实验报告可以对应 `agent_exp_guide.md` 的结构：

- 研究背景：内容真实性、内容溯源、图像取证
- 论文介绍：Rao2016 用 CNN 检测 splicing / copy-move
- 方法原理：patch 采样、CNN 特征、SRM 残差思想、分类器
- 简化复现：本项目使用轻量 CNN 直接二分类
- 实验结果：loss 曲线、accuracy 曲线、混淆矩阵、预测样例
- 分析总结：简化版和原论文的差距，以及可改进方向
