#!/usr/bin/env python3
"""
Prepare fred drone dataset for YOLO training.

Usage:
    # 使用指定的划分文件 (推荐)
    python prepare_fred_dataset.py --train train.txt --val val.txt --test test.txt

    # 仅使用 train 和 val (无 test)
    python prepare_fred_dataset.py --train train.txt --val val.txt

    # 自动扫描并划分 (80/20)
    python prepare_fred_dataset.py

    # 自动扫描并限制样本数
    python prepare_fred_dataset.py --samples 1000

Steps:
    1. Read image paths from txt files (or scan dataset directory)
    2. Find corresponding YOLO annotations from RGB_YOLO folders
    3. Create dataset directory structure with symlinks
"""

import argparse
from pathlib import Path


# Dataset source configuration
DATASET_ROOT = Path("/mnt/data/datasets/fred")
IMAGE_SIZE = (1280, 720)  # Image dimensions (width, height)


def scan_all_images() -> list[str]:
    """
    Scan dataset directory and return all image paths with annotations.

    Returns:
        List of image paths that have corresponding annotations
    """
    all_images = []

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
            if ann_path.exists():
                all_images.append(str(img_path))

        if (i + 1) % 10 == 0:
            print(f"  Scanned {i + 1}/{len(folders)} folders, {len(all_images)} images")

    print(f"Total: {len(all_images)} images with annotations")
    return all_images


def read_split_file(txt_path: str) -> list[tuple[str, str]]:
    """
    Read image paths and annotations from a split file.

    Supports two formats:
    1. With annotations: /path/to/image.jpg x1,y1,x2,y2,class [x1,y1,x2,y2,class ...]
    2. Image only: /path/to/image.jpg

    Args:
        txt_path: Path to txt file

    Returns:
        List of (image_path, yolo_annotation) tuples
    """
    path = Path(txt_path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {txt_path}")

    data = []
    img_w, img_h = IMAGE_SIZE

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(' ', 1)  # Split into path and optional annotations
            img_path = parts[0]

            if len(parts) > 1:
                # Format: /path/to/image.jpg x1,y1,x2,y2,class [...]
                # Convert absolute coords to YOLO format
                annotations = parts[1].split()
                yolo_anns = []
                for ann in annotations:
                    coords = ann.split(',')
                    if len(coords) != 5:
                        continue
                    try:
                        x1, y1, x2, y2, cls = map(float, coords)
                    except ValueError:
                        continue

                    # Skip invalid annotations (negative coords)
                    if x1 < 0 or y1 < 0:
                        continue

                    # Convert to normalized center format
                    x_center = (x1 + x2) / 2 / img_w
                    y_center = (y1 + y2) / 2 / img_h
                    w = (x2 - x1) / img_w
                    h = (y2 - y1) / img_h

                    # Clamp values to [0, 1]
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    w = max(0, min(1, w))
                    h = max(0, min(1, h))

                    yolo_anns.append(f"{int(cls)} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

                data.append((img_path, '\n'.join(yolo_anns)))
            else:
                # Format: /path/to/image.jpg (no annotations in file)
                # Will read from RGB_YOLO folder
                data.append((img_path, None))

    # Count how many have inline annotations
    inline_count = sum(1 for _, ann in data if ann is not None)
    print(f"Read {len(data)} images from {txt_path} ({inline_count} with inline annotations)")
    return data


def get_annotation(img_path: str) -> str:
    """
    Get YOLO annotation for an image from RGB_YOLO folder.
    Only used when annotations are not provided in the split file.

    Args:
        img_path: Path to image file

    Returns:
        YOLO format annotation string
    """
    img = Path(img_path)
    # Find RGB_YOLO folder in same parent directory
    folder = img.parent.parent  # PADDED_RGB -> folder_num
    ann_path = folder / "RGB_YOLO" / f"{img.stem}.txt"

    if not ann_path.exists():
        return ""

    with open(ann_path, 'r') as f:
        return f.read().strip()


def prepare_dataset_from_splits(
    train_txt: str = None,
    val_txt: str = None,
    test_txt: str = None,
    output_dir: str = "/mnt/data/datasets/fred_drone",
):
    """
    Prepare dataset using specified split files.
    Ensures the split is exactly as specified in the txt files.

    Args:
        train_txt: Path to train split file
        val_txt: Path to val split file
        test_txt: Path to test split file (optional)
        output_dir: Output directory for the prepared dataset
    """
    output_path = Path(output_dir)

    # Read split files
    train_images = read_split_file(train_txt) if train_txt else []
    val_images = read_split_file(val_txt) if val_txt else []
    test_images = read_split_file(test_txt) if test_txt else []

    if not train_images and not val_images:
        raise ValueError("At least one of train_txt or val_txt must be provided")

    print(f"\nSplit sizes: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")

    # Check for overlaps (compare image paths only)
    train_paths = set(img_path for img_path, _ in train_images)
    val_paths = set(img_path for img_path, _ in val_images)
    test_paths = set(img_path for img_path, _ in test_images)

    if train_paths & val_paths:
        print(f"Warning: {len(train_paths & val_paths)} images in both train and val")
    if train_paths & test_paths:
        print(f"Warning: {len(train_paths & test_paths)} images in both train and test")
    if val_paths & test_paths:
        print(f"Warning: {len(val_paths & test_paths)} images in both val and test")

    # Create directory structure
    dirs = {}
    if train_images:
        dirs['train'] = {
            'images': output_path / "images" / "train",
            'labels': output_path / "labels" / "train",
        }
    if val_images:
        dirs['val'] = {
            'images': output_path / "images" / "val",
            'labels': output_path / "labels" / "val",
        }
    if test_images:
        dirs['test'] = {
            'images': output_path / "images" / "test",
            'labels': output_path / "labels" / "test",
        }

    for split_name, split_dirs in dirs.items():
        for d in split_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    def process_split(split_name: str, image_data: list[tuple[str, str]]) -> int:
        """Process a single split and return count of successful samples.

        Args:
            split_name: Name of the split (train/val/test)
            image_data: List of (image_path, yolo_annotation) tuples

        Returns:
            Count of successful samples
        """
        if not image_data:
            return 0

        print(f"\nProcessing {split_name} set...")
        images_dir = dirs[split_name]['images']
        labels_dir = dirs[split_name]['labels']

        count = 0
        missing = 0
        for img_path, yolo_ann in image_data:
            # Use pre-parsed annotation or read from RGB_YOLO folder
            if not yolo_ann:
                yolo_ann = get_annotation(img_path)

            if not yolo_ann:
                missing += 1
                continue

            # Create symlink
            src = Path(img_path)
            dst = images_dir / src.name

            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)

            # Write label
            label_path = labels_dir / f"{src.stem}.txt"
            label_path.write_text(yolo_ann)
            count += 1

        print(f"  Created {count} samples, {missing} missing annotations")
        return count

    # Process each split
    train_count = process_split('train', train_images)
    val_count = process_split('val', val_images)
    test_count = process_split('test', test_images)

    # Create dataset YAML
    yaml_content = f"""# Fred Drone Dataset
# Generated by prepare_fred_dataset.py

path: {output_path}
train: images/train
val: images/val
test: images/test

names:
  0: drone
"""
    yaml_path = output_path / "fred_drone.yaml"
    yaml_path.write_text(yaml_content)
    print(f"\nCreated dataset config: {yaml_path}")

    print("\nDataset preparation complete!")
    print(f"  Train: {train_count}")
    print(f"  Val: {val_count}")
    print(f"  Test: {test_count}")
    print(f"  Config: {yaml_path}")


def prepare_dataset_auto(
    output_dir: str = "/mnt/data/datasets/fred_drone",
    train_ratio: float = 0.8,
    seed: int = 42,
    max_samples: int = None,
):
    """
    Prepare dataset by auto-scanning and splitting.

    Args:
        output_dir: Output directory for the prepared dataset
        train_ratio: Ratio for train split
        seed: Random seed for reproducibility
        max_samples: Maximum samples to use (None for all)
    """
    import random
    random.seed(seed)

    # Scan all images
    all_images = scan_all_images()

    # Limit samples if specified
    if max_samples and max_samples < len(all_images):
        random.shuffle(all_images)
        all_images = all_images[:max_samples]
        print(f"Limited to {max_samples} samples")

    # Shuffle and split
    random.shuffle(all_images)
    split_idx = int(len(all_images) * train_ratio)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    # Write temporary split files
    Path("train_auto.txt").write_text('\n'.join(train_images))
    Path("val_auto.txt").write_text('\n'.join(val_images))

    print(f"Auto-split: train={len(train_images)}, val={len(val_images)}")

    # Use the split file method
    prepare_dataset_from_splits(
        train_txt="train_auto.txt",
        val_txt="val_auto.txt",
        output_dir=output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare Fred drone dataset for YOLO training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use specified split files (recommended)
  python prepare_fred_dataset.py --train train.txt --val val.txt --test test.txt

  # Use only train and val
  python prepare_fred_dataset.py --train train.txt --val val.txt

  # Auto-scan and split (80/20)
  python prepare_fred_dataset.py

  # Auto-scan with sample limit
  python prepare_fred_dataset.py --samples 1000
"""
    )
    parser.add_argument("--train", type=str, help="Train split file (image paths, one per line)")
    parser.add_argument("--val", type=str, help="Val split file (image paths, one per line)")
    parser.add_argument("--test", type=str, help="Test split file (image paths, one per line)")
    parser.add_argument("--output", "-o", type=str, default="/mnt/data/datasets/fred_drone",
                        help="Output dataset directory (default: /mnt/data/datasets/fred_drone)")

    # Auto-split options (when split files not provided)
    parser.add_argument("--samples", "-n", type=int, default=None,
                        help="Maximum samples to use (auto mode only)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Train/val split ratio (auto mode only, default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (auto mode only, default: 42)")

    args = parser.parse_args()

    if args.train or args.val:
        # Use specified split files
        prepare_dataset_from_splits(
            train_txt=args.train,
            val_txt=args.val,
            test_txt=args.test,
            output_dir=args.output,
        )
    else:
        # Auto-scan and split
        prepare_dataset_auto(
            output_dir=args.output,
            train_ratio=args.train_ratio,
            seed=args.seed,
            max_samples=args.samples,
        )
