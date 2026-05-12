from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from .data import CLASS_NAMES, IMAGE_SIZE, build_transforms
from .model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict whether an image is authentic or forged.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    # 推理时切到 eval，确保 BatchNorm 使用训练好的统计量。
    model.eval()

    transform = build_transforms(image_size=args.image_size, train=False)
    image = Image.open(args.image).convert("RGB")
    # unsqueeze(0) 添加 batch 维度，让单张图片也能走模型的批处理接口。
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred = int(probs.argmax().item())

    print(f"image: {args.image}")
    print(f"prediction: {CLASS_NAMES[pred]}")
    print(f"confidence: {probs[pred].item():.4f}")
    print("probabilities:")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {probs[idx].item():.4f}")


if __name__ == "__main__":
    main()
