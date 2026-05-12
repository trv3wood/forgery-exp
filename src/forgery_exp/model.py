from __future__ import annotations

import torch
from torch import nn


class SimpleForgeryCNN(nn.Module):
    """Lightweight patch CNN inspired by Rao2016's patch-based classifier."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        # 小型卷积网络提取局部纹理痕迹，适合这个合成篡改实验的低成本 baseline。
        self.features = nn.Sequential(
            nn.Conv2d(3, 30, kernel_size=5, padding=2),
            nn.BatchNorm2d(30),
            nn.ReLU(inplace=True),
            nn.Conv2d(30, 30, kernel_size=5, padding=2),
            nn.BatchNorm2d(30),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(30, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 自适应池化把任意空间尺寸压到 1x1，分类层只关心通道特征。
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model(num_classes: int = 2) -> SimpleForgeryCNN:
    return SimpleForgeryCNN(num_classes=num_classes)
