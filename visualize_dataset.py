#!/usr/bin/env python3
"""
Visualize Fred drone dataset annotations.

Usage:
    # Visualize random samples from train set
    python visualize_dataset.py

    # Visualize specific split
    python visualize_dataset.py --split val

    # Visualize specific number of samples
    python visualize_dataset.py --num 10

    # Save visualizations to directory
    python visualize_dataset.py --save --output ./vis_results
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


DATASET_PATH = Path("/mnt/data/datasets/fred_drone")
IMAGE_SIZE = (1280, 720)  # width, height


def yolo_to_bbox(yolo_line: str, img_w: int, img_h: int) -> tuple:
    """Convert YOLO format to bbox coordinates."""
    parts = yolo_line.strip().split()
    if len(parts) < 5:
        return None

    cls = int(parts[0])
    xc, yc, w, h = map(float, parts[1:5])

    x1 = int((xc - w/2) * img_w)
    y1 = int((yc - h/2) * img_h)
    x2 = int((xc + w/2) * img_w)
    y2 = int((yc + h/2) * img_h)

    return (cls, x1, y1, x2, y2)


def visualize_image(
    img_path: Path,
    label_path: Path,
    show: bool = True,
    save_path: Path = None,
):
    """Visualize image with annotations."""
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Warning: Cannot read image {img_path}")
        return

    img_h, img_w = img.shape[:2]

    # Read labels
    if not label_path.exists():
        print(f"Warning: Label file not found {label_path}")
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Draw bboxes
    colors = {
        0: (0, 255, 0),  # Green for drone
    }

    for line in lines:
        bbox = yolo_to_bbox(line, img_w, img_h)
        if bbox is None:
            continue

        cls, x1, y1, x2, y2 = bbox
        color = colors.get(cls, (255, 255, 255))

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"drone" if cls == 0 else f"class_{cls}"
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw image path
    cv2.putText(img, str(img_path.name), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Save or show
    if save_path:
        cv2.imwrite(str(save_path), img)
        print(f"Saved: {save_path}")

    if show:
        cv2.imshow("Annotation", img)
        print(f"Showing: {img_path.name}")
        print("Press any key to continue, 'q' to quit...")
        key = cv2.waitKey(0)
        if key == ord('q'):
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Visualize Fred drone dataset")
    parser.add_argument(
        "--split", type=str, default="train",
        choices=["train", "val", "test"],
        help="Dataset split to visualize"
    )
    parser.add_argument(
        "--num", "-n", type=int, default=5,
        help="Number of images to visualize"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save visualizations instead of showing"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="./vis_results",
        help="Output directory for saved visualizations"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)

    # Get image and label directories
    images_dir = DATASET_PATH / "images" / args.split
    labels_dir = DATASET_PATH / "labels" / args.split

    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return

    # Get all images
    images = list(images_dir.glob("*.jpg"))
    if not images:
        print(f"Error: No images found in {images_dir}")
        return

    print(f"Found {len(images)} images in {args.split} set")

    # Sample images
    if args.num < len(images):
        images = random.sample(images, args.num)

    # Create output directory
    if args.save:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving visualizations to {output_dir}")

    # Visualize
    for i, img_path in enumerate(images):
        label_path = labels_dir / f"{img_path.stem}.txt"

        save_path = None
        if args.save:
            save_path = Path(args.output) / f"{img_path.stem}_vis.jpg"

        if not visualize_image(
            img_path,
            label_path,
            show=not args.save,
            save_path=save_path,
        ):
            break

    cv2.destroyAllWindows()
    print("\nDone!")


if __name__ == "__main__":
    main()
