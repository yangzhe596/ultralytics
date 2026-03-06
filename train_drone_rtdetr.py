#!/usr/bin/env python3
"""
Train RT-DETR on Fred Drone dataset.

RT-DETR is a real-time detection transformer that provides excellent
accuracy-speed trade-off. Available models:
- rtdetr-l.yaml: RT-DETR-L (large)
- rtdetr-x.yaml: RT-DETR-X (extra large)
- rtdetr-resnet50.yaml: RT-DETR with ResNet-50 backbone
- rtdetr-resnet101.yaml: RT-DETR with ResNet-101 backbone

Usage:
    python train_drone_rtdetr.py
"""

from ultralytics import RTDETR


def train_rtdetr():
    """Train RT-DETR-L on drone detection dataset."""

    # Load RT-DETR-L model
    model = RTDETR("rtdetr-l.yaml")

    # Train the model
    results = model.train(
        data="/mnt/data/datasets/fred_drone/fred_drone.yaml",
        epochs=15,
        imgsz=640,
        batch=8,  # RT-DETR needs smaller batch due to larger memory usage
        device=0,
        workers=4,
        project="runs/detect",
        name="drone_rtdetr_l",
        exist_ok=True,
        pretrained=False,  # Train from scratch
        optimizer="auto",
        verbose=True,
        seed=42,
    )

    print(f"\nTraining complete!")
    print(f"Results saved to: runs/detect/drone_rtdetr_l")

    # Validate the model
    print("\nRunning validation...")
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    return results, metrics


if __name__ == "__main__":
    train_rtdetr()
