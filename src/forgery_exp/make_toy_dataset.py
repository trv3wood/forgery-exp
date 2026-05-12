from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm


def smooth_noise(size: int, rng: np.random.Generator) -> Image.Image:
    # 从低分辨率噪声放大并模糊，生成带有平滑纹理的“原始图像”背景。
    small = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    img = Image.fromarray(small, mode="RGB").resize((size, size), Image.Resampling.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))

    draw = ImageDraw.Draw(img, mode="RGBA")
    # 叠加半透明几何块，让图片有可复制粘贴的局部结构。
    for _ in range(rng.integers(3, 8)):
        x0 = int(rng.integers(0, size - 24))
        y0 = int(rng.integers(0, size - 24))
        x1 = int(min(size, x0 + rng.integers(20, 72)))
        y1 = int(min(size, y0 + rng.integers(20, 72)))
        color = tuple(int(v) for v in rng.integers(0, 256, size=3)) + (int(rng.integers(35, 90)),)
        if rng.random() < 0.5:
            draw.rectangle([x0, y0, x1, y1], fill=color)
        else:
            draw.ellipse([x0, y0, x1, y1], fill=color)
    return img.filter(ImageFilter.GaussianBlur(radius=0.4))


def make_forged(base: Image.Image, donor: Image.Image, rng: np.random.Generator) -> Image.Image:
    size = base.size[0]
    forged = base.copy()

    # 随机裁一块源区域，再粘贴到目标位置，模拟 copy-move 或拼接伪造。
    patch_size = int(rng.integers(size // 5, size // 3))
    sx = int(rng.integers(0, size - patch_size))
    sy = int(rng.integers(0, size - patch_size))
    tx = int(rng.integers(0, size - patch_size))
    ty = int(rng.integers(0, size - patch_size))

    source = base if rng.random() < 0.5 else donor
    patch = source.crop((sx, sy, sx + patch_size, sy + patch_size))

    if rng.random() < 0.35:
        patch = patch.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < 0.35:
        patch = patch.filter(ImageFilter.SHARPEN)

    forged.paste(patch, (tx, ty))

    # 加一条很淡的边界痕迹，让 toy 数据既可学习，又不会完全靠明显白框分类。
    draw = ImageDraw.Draw(forged, mode="RGBA")
    draw.rectangle(
        [tx, ty, tx + patch_size - 1, ty + patch_size - 1],
        outline=(255, 255, 255, int(rng.integers(18, 42))),
        width=1,
    )
    return forged


def write_split(output: Path, split: str, count: int, size: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # 目录名必须和训练代码中的 CLASS_NAMES 对齐，ImageFolder 会用它们生成标签。
    authentic_dir = output / split / "authentic"
    forged_dir = output / split / "forged"
    authentic_dir.mkdir(parents=True, exist_ok=True)
    forged_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(count), desc=f"Generating {split}"):
        base = smooth_noise(size=size, rng=rng)
        donor = smooth_noise(size=size, rng=rng)
        forged = make_forged(base=base, donor=donor, rng=rng)

        base.save(authentic_dir / f"authentic_{idx:04d}.png")
        forged.save(forged_dir / f"forged_{idx:04d}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small synthetic forgery dataset.")
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--train", type=int, default=200)
    parser.add_argument("--val", type=int, default=50)
    parser.add_argument("--test", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_split(args.output, "train", args.train, args.size, args.seed)
    write_split(args.output, "val", args.val, args.size, args.seed + 1)
    write_split(args.output, "test", args.test, args.size, args.seed + 2)


if __name__ == "__main__":
    main()
