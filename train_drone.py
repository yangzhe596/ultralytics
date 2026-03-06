#!/usr/bin/env python3
"""
Train YOLO12 on Fred Drone dataset.
"""

from ultralytics import YOLO


def train_drone():
    """Train YOLO12n on drone detection dataset."""

    # Load YOLO12n model (smallest, fastest)
    model = YOLO("yolo12n.yaml")  # Build from YAML

    # Train the model
    results = model.train(
        data="/mnt/data/datasets/fred_drone/fred_drone.yaml",
        epochs=15,
        imgsz=640,
        batch=16,
        device=0,  # GPU 0
        workers=4,
        project="runs/detect",
        name="drone_yolo12n",
        exist_ok=True,
        pretrained=False,  # Train from scratch
        optimizer="auto",
        verbose=True,
        seed=42,
    )

    print(f"\nTraining complete!")
    print(f"Results saved to: runs/detect/drone_yolo12n")

    # Validate the model
    print("\nRunning validation...")
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    return results, metrics


if __name__ == "__main__":
    train_drone()
