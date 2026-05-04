我看完云盘里的 3 个文件后，建议你这次作业不要死磕完整复现 Rao2016，而是做成：

**“Rao2016 图像篡改检测论文讲解 + 简化版 CNN 代码复现实验”**

这样最稳：既贴合 Lecture 1 第16-19页“内容真实性、内容溯源、图像与视频取证”的主题，又不会因为数据集、Caffe、SVM、SRM 滤波器完整复现卡死。

---

## 1. 作业主题怎么对齐课程要求

Lecture 1 第16-19页主要讲“信息内容安全研究”。其中第17页明确提到：

内容真实吗？包括假新闻、深度伪造、AIGC安全等；内容从哪里来？包括数字水印、感知哈希、图像与视频取证、隐写分析等。

所以 Rao2016 这篇论文非常对题，因为它做的是：

> 用 CNN 检测图像拼接和复制-移动篡改。

论文摘要也说明，它使用 CNN 自动学习 RGB 图像中的层次特征，用于 splicing 和 copy-move 检测，并用 SRM 高通滤波器初始化第一层卷积核，最后用 SVM 分类。

---

## 2. 三个文件的定位

### A. Lecture 1 绪论.pdf

它是作业选题依据。重点看第16-19页：

第16页是内容智能分析与识别，包括文本、多媒体、跨模态理解。
第17页是虚假信息、AIGC治理、内容溯源、图像与视频取证。
第18页是传播分析和隐私保护。
第19页是内容合规与治理机制。

你的作业可以归到：

**内容从哪里来？——内容溯源与数字取证——图像篡改检测。**

---

### B. Rao2016Deep 论文

这是你的主论文。核心方法如下：

1. 从篡改边界附近采样正样本 patch，从真实图像中随机采样负样本 patch。
2. 训练 CNN 学习局部篡改痕迹。
3. 用滑动窗口扫描整张测试图，提取 patch 特征。
4. 对 patch 特征做 mean/max pooling，得到整图特征。
5. 用 SVM 做 authentic/forged 二分类。

网络结构也很清楚：输入是 `128×128×3` patch，8个卷积层、2个池化层、1个全连接层、2分类 softmax。

最有讲解价值的创新点是：

**第一层卷积核不是随机初始化，而是用 SRM 的 30 个高通滤波器初始化。**

这样可以压制图像内容本身，突出篡改留下的细微残差。论文解释说，建模 residual 而不是像素值，可以抑制图像内容，提高篡改痕迹相对信号强度。

实验数据集包括 CASIA v1.0、CASIA v2.0 和 Columbia DVMM。论文中 CASIA v2.0 有 7491 张真实图和 5123 张篡改图，包含 splicing 和 copy-move。

# 推荐实验方案：Rao2016 简化复现

## 实验目标

复现一个简化版图像篡改检测流程：

> 输入一张图片，判断它是真实图像还是被拼接 / 复制移动篡改过的图像。

不要完整复现论文的 Caffe + SRM + SVM 全流程。你可以做成 PyTorch 版本：

```text
图像数据集
   ↓
裁剪 128×128 patch
   ↓
CNN 二分类模型
   ↓
输出 authentic / forged
   ↓
准确率、loss 曲线、混淆矩阵
```

这已经足够作为“算法代码复现”。

---

## 实验数据怎么准备

### 方案一：用公开篡改数据集，最贴论文

论文用的是：

```text
CASIA v1.0
CASIA v2.0
Columbia DVMM
```

论文明确写到 CASIA v1.0 / v2.0 都包含 splicing 和 copy-move 篡改图像。

你的目录可以整理成：

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

如果 CASIA 下载麻烦，就用第二种。

---

## 实验步骤

### Step 1：环境准备

建议用 PyTorch：

```bash
conda create -n forgery python=3.10
conda activate forgery

pip install torch torchvision pillow matplotlib scikit-learn tqdm opencv-python
```

---

### Step 2：数据预处理

每张图片统一处理：

```text
Resize 到 256×256
随机裁剪 128×128 patch
归一化到 [0,1]
标签：
  authentic = 0
  forged = 1
```

Rao2016 原文用的 patch 大小就是 `128×128×3`。

---

### Step 3：模型设计

你可以写一个简化 CNN：

```text
Conv2d(3, 30, 5) + ReLU
Conv2d(30, 30, 5) + ReLU + MaxPool
Conv2d(30, 16, 3) + ReLU
Conv2d(16, 16, 3) + ReLU + MaxPool
AdaptiveAvgPool2d
Linear -> 2分类
```

报告里说明：

> 原论文使用 8 个卷积层、2 个池化层和 1 个全连接层。本实验为了降低复现难度，保留“patch-based CNN 二分类”的核心思想，使用轻量 CNN 完成真实/篡改分类。

---

### Step 4：训练

核心训练指标：

```text
train loss
val accuracy
test accuracy
confusion matrix
```

建议训练参数：

```text
epochs = 20
batch_size = 32
optimizer = Adam
learning_rate = 1e-3
loss = CrossEntropyLoss
```

---

### Step 5：可视化输出

报告里至少放这几张图：

```text
1. 数据样例：真实图 vs 篡改图
2. 训练 loss 曲线
3. 验证 accuracy 曲线
4. 混淆矩阵
5. 若干预测结果：图片 + 真实标签 + 预测标签
```

这样报告会很完整。

---

# 报告结构建议

可以直接按这个写：

```text
1. 研究背景
   - 信息内容安全关注内容真实性、内容溯源和数字取证
   - 图像拼接和复制移动是常见篡改方式

2. 论文介绍
   - 论文：A Deep Learning Approach to Detection of Splicing and Copy-Move Forgeries in Images
   - 任务：检测 authentic / forged 图像
   - 方法：CNN 提取 patch 特征 + 特征融合 + SVM 分类

3. 方法原理
   - patch 采样
   - CNN 特征学习
   - SRM 高通滤波器初始化
   - 滑动窗口特征提取
   - pooling 特征融合
   - SVM / 分类器判断真假

4. 简化复现实验设计
   - 数据集来源
   - 数据预处理
   - 网络结构
   - 训练参数

5. 实验结果
   - loss 曲线
   - accuracy 曲线
   - 混淆矩阵
   - 预测样例

6. 分析与总结
   - CNN 能学习到局部篡改痕迹
   - 高通残差思想可以突出篡改边界
   - 简化实验与原论文差距：没有完整 SRM 初始化、没有完整滑动窗口 + SVM
   - 后续可改进：加入 SRM 滤波器、使用 CASIA v2.0、做篡改区域定位
```

---

# 最小可交付版本

如果时间紧，做到这个程度就可以交：

```text
1. 讲解 Rao2016 论文核心思想
2. 用 PyTorch 写一个 CNN
3. 准备 authentic / forged 两类图片
4. 训练二分类模型
5. 输出准确率、loss 曲线、预测样例
6. 在报告中说明这是简化复现
```

不建议把 GAN-SIN.py 当主实验。它可以放在附录或课堂扩展里，说明：

