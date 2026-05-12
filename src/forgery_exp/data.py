from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGE_SIZE = 128
CLASS_NAMES = ["authentic", "forged"]


def build_transforms(image_size: int = IMAGE_SIZE, train: bool = False) -> transforms.Compose:
    # 先放大再裁剪：训练集用随机裁剪增强鲁棒性，验证/测试集用中心裁剪保证结果稳定。
    steps: list[object] = [
        transforms.Resize((image_size + 32, image_size + 32)),
    ]

    if train:
        steps.extend(
            [
                transforms.RandomCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
    else:
        steps.append(transforms.CenterCrop((image_size, image_size)))

    steps.extend(
        [
            transforms.ToTensor(),
            # 将像素从 [0, 1] 映射到约 [-1, 1]，与训练和预测阶段保持同一输入分布。
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(steps)


def build_dataset(data_dir: str | Path, split: str, image_size: int = IMAGE_SIZE) -> datasets.ImageFolder:
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing dataset split: {split_dir}")

    dataset = datasets.ImageFolder(
        root=split_dir,
        transform=build_transforms(image_size=image_size, train=(split == "train")),
    )

    # ImageFolder 会按文件夹名字排序生成标签，这里固定标签顺序，避免训练和预测类别错位。
    expected = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    if dataset.class_to_idx != expected:
        raise ValueError(
            f"Expected class folders {expected}, got {dataset.class_to_idx}. "
            "Please use authentic/ and forged/ folders."
        )
    return dataset


def build_loader(
    data_dir: str | Path,
    split: str,
    image_size: int = IMAGE_SIZE,
    batch_size: int = 32,
    num_workers: int = 0,
) -> DataLoader:
    dataset = build_dataset(data_dir=data_dir, split=split, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        # 只打乱训练集；验证/测试集保持顺序，便于复现实验和排查样本。
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
