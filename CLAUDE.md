# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build, Test, and Development Commands

```bash
# Install in editable mode for development
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Run all tests (slow tests excluded by default)
pytest

# Run tests including slow ones
pytest --slow

# Run a single test file
pytest tests/test_python.py

# Run a specific test
pytest tests/test_python.py::test_function_name

# Run with coverage
pytest --cov=ultralytics --cov-report=html

# CLI usage examples
yolo predict model=yolo26n.pt source='image.jpg'
yolo train model=yolo26n.pt data=coco8.yaml epochs=10
yolo val model=yolo26n.pt data=coco8.yaml
yolo export model=yolo26n.pt format=onnx

# Code formatting
ruff format .
yapf -i <file>

# Spell check
codespell
```

## Architecture Overview

Ultralytics YOLO is a computer vision library supporting object detection, instance segmentation, image classification, pose estimation, and oriented bounding boxes (OBB).

### Core Components

- **`ultralytics/models/`**: Model definitions. Each model (YOLO, YOLOWorld, YOLOE, NAS, SAM, FastSAM, RTDETR) has a `model.py` with the class and task-specific trainer/validator/predictor mappings.

- **`ultralytics/engine/`**: Core framework classes:
  - `model.py`: Base `Model` class - handles training, validation, prediction, export, benchmarking
  - `trainer.py`: `BaseTrainer` - training loop implementation
  - `validator.py`: `BaseValidator` - validation logic
  - `predictor.py`: `BasePredictor` - inference pipeline
  - `exporter.py`: Model export to ONNX, TensorRT, CoreML, etc.
  - `results.py`: `Results` class for prediction outputs

- **`ultralytics/nn/`**: Neural network architecture:
  - `tasks.py`: Model building (`DetectionModel`, `SegmentationModel`, etc.) and weight loading
  - `modules/`: Layer implementations (Conv, C2f, Detect, Segment, etc.)
  - `autobackend.py`: Multi-format inference backend

- **`ultralytics/data/`**: Data handling:
  - `dataset.py`: Dataset classes for different tasks
  - `augment.py`: Data augmentation transforms
  - `build.py`: Data loader construction
  - `utils.py`: Dataset utilities

- **`ultralytics/cfg/`**: Configuration:
  - `__init__.py`: CLI entrypoint (`entrypoint()` function)
  - `default.yaml`: Default hyperparameters
  - `models/`: Model architecture YAMLs
  - `datasets/`: Dataset configuration YAMLs

### Model Instantiation Flow

1. User creates model: `YOLO("yolo26n.pt")` or `YOLO("yolo26n.yaml")`
2. `Model.__init__()` determines source type (checkpoint, YAML config, HUB, Triton)
3. `_load()` (for .pt) or `_new()` (for .yaml) initializes the model
4. `task_map` property maps task to model/trainer/validator/predictor classes
5. Methods like `.train()`, `.val()`, `.predict()` use `_smart_load()` to get appropriate classes

### Tasks

Five tasks are supported: `detect`, `segment`, `classify`, `pose`, `obb`. Task-specific implementations are in `ultralytics/models/yolo/<task>/`.

### Key Patterns

- **Task mapping**: Each model class defines `task_map` returning a dict mapping task names to model/trainer/validator/predictor classes
- **Lazy imports**: Model classes (YOLO, SAM, etc.) are lazily imported via `__getattr__` in `__init__.py`
- **Callback system**: Training/validation supports custom callbacks via `callbacks` module
- **Auto-backend**: `AutoBackend` class handles inference across PyTorch, ONNX, TensorRT, CoreML, etc.

### Adding a New YOLO Version

1. Add model YAMLs in `ultralytics/cfg/models/<version>/`
2. Implement new modules in `ultralytics/nn/modules/`
3. Register modules in `ultralytics/nn/tasks.py` imports and `guess_model_task()`
4. Update `TASK2MODEL` in `ultralytics/cfg/__init__.py`
