#!/usr/bin/env python3
"""
Generate split files (train.txt, val.txt, test.txt) from Fred drone dataset.

Scans all images with valid annotations and generates split files with
inline annotations in format: path x1,y1,x2,y2,class

Usage:
    # Generate splits with default 80/10/10 ratio
    python generate_splits.py

    # Generate with custom sample size
    python generate_splits.py --samples 1000

    # Generate with custom split ratio
    python generate_splits.py --train-ratio 0.8 --val-ratio 0.1
"""

import argparse
import random
from pathlib import Path


# Dataset configuration
DATASET_ROOT = Path("/mnt/data/datasets/fred")
IMAGE_SIZE = (1280, 720)  # width, height


def scan_dataset() -> list[tuple[str, str]]:
    """
    Scan dataset directory and return all images with annotations.

    Returns:
        List of (image_path, inline_annotation) tuples
    """
    all_data = []
    img_w, img_h = IMAGE_SIZE

    folders = sorted(
        [f for f in DATASET_ROOT.iterdir() if f.is_dir() and f.name.isdigit()],
        key=lambda x: int(x.name)
    )

    print(f"Scanning {len(folders)} folders...")

    for i, folder in enumerate(folders):
        padded_rgb = folder / "PADDED_RGB"
        rgb_yolo = folder / "RGB_YOLO"

        if not padded_rgb.exists() or not rgb_yolo.exists():
            continue

        for img_path in sorted(padded_rgb.glob("*.jpg")):
            ann_path = rgb_yolo / f"{img_path.stem}.txt"
            if not ann_path.exists():
                continue

            # Read YOLO annotation and convert to inline format
            with open(ann_path, 'r') as f:
                yolo_lines = f.readlines()

            if not yolo_lines:
                continue

            ann_parts = []
            for line in yolo_lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cls, xc, yc, w, h = map(float, parts[:5])
                        # Convert normalized center format to absolute coords
                        x1 = int((xc - w/2) * img_w)
                        y1 = int((yc - h/2) * img_h)
                        x2 = int((xc + w/2) * img_w)
                        y2 = int((yc + h/2) * img_h)
                        ann_parts.append(f"{x1},{y1},{x2},{y2},{int(cls)}")
                    except ValueError:
                        continue

            if ann_parts:
                all_data.append((str(img_path), ' '.join(ann_parts)))

        if (i + 1) % 10 == 0:
            print(f"  Scanned {i + 1}/{len(folders)} folders, {len(all_data)} images")

    print(f"Total: {len(all_data)} images with annotations")
    return all_data


def generate_splits(
    all_data: list[tuple[str, str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    max_samples: int = None,
    output_dir: str = ".",
):
    """
    Generate train.txt, val.txt, test.txt split files.

    Args:
        all_data: List of (image_path, annotation) tuples
        train_ratio: Ratio for train split
        val_ratio: Ratio for val split (remaining goes to test)
        seed: Random seed
        max_samples: Maximum samples to use (None for all)
        output_dir: Output directory for split files
    """
    random.seed(seed)

    # Shuffle data
    data = all_data.copy()
    random.shuffle(data)

    # Limit samples if specified
    if max_samples and max_samples < len(data):
        data = data[:max_samples]
        print(f"Limited to {max_samples} samples")

    # Calculate split sizes
    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    print(f"\nSplit sizes: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # Write split files
    output_path = Path(output_dir)

    def write_split(filename: str, data: list[tuple[str, str]]):
        filepath = output_path / filename
        with open(filepath, 'w') as f:
            for img_path, ann in data:
                f.write(f"{img_path} {ann}\n")
        print(f"  Created {filepath} ({len(data)} samples)")

    write_split("train.txt", train_data)
    write_split("val.txt", val_data)
    write_split("test.txt", test_data)

    print("\nDone! Split files generated.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate train/val/test split files from Fred drone dataset"
    )
    parser.add_argument(
        "--samples", "-n", type=int, default=None,
        help="Maximum samples to use (default: all)"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="Train split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="Val split ratio (default: 0.1, remaining goes to test)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=".",
        help="Output directory (default: current directory)"
    )

    args = parser.parse_args()

    # Scan dataset
    all_data = scan_dataset()

    if not all_data:
        print("Error: No valid images found!")
        return

    # Generate splits
    generate_splits(
        all_data=all_data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_samples=args.samples,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
