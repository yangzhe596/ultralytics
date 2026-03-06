# Fred Drone 数据集 YOLO12 训练指南

## 快速开始

```bash
# 激活环境
conda activate yolo

# 1. 准备数据集划分文件 (train.txt, val.txt, test.txt)
# 格式: /path/to/image.jpg x1,y1,x2,y2,class [x1,y1,x2,y2,class ...]

# 2. 生成数据集
python prepare_fred_dataset.py --train train.txt --val val.txt --test test.txt

# 3. 训练模型
python train_drone.py

# 4. 测试模型
yolo predict model=runs/detect/drone_yolo12n/weights/best.pt source=/mnt/data/datasets/fred_drone/images/test
```

## 环境配置

### 创建 Conda 环境

```bash
conda create -n yolo python=3.10 -y
conda activate yolo

# 安装 PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装 ultralytics (开发模式)
pip install -e ".[dev]"
```

### 验证环境

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## 数据准备

### 数据划分文件格式

划分文件支持两种格式：

**格式1: 带标注 (推荐)**
```text
# train.txt
/mnt/data/datasets/fred/0/PADDED_RGB/Video_0_16_03_03.jpg 816,305,854,336,0
/mnt/data/datasets/fred/1/PADDED_RGB/Video_1_16_04_12.jpg 755,202,834,243,0 446,367,520,400,0
```
- 每行格式: `图片路径 x1,y1,x2,y2,class [x1,y1,x2,y2,class ...]`
- 多个目标用空格分隔
- 坐标为绝对像素坐标

**格式2: 仅路径**
```text
/mnt/data/datasets/fred/0/PADDED_RGB/Video_0_16_03_03.jpg
```
- 脚本会从 `RGB_YOLO` 文件夹读取对应标注

### 生成数据集

```bash
# 使用指定的划分文件 (推荐)
python prepare_fred_dataset.py --train train.txt --val val.txt --test test.txt

# 仅使用 train 和 val (无 test)
python prepare_fred_dataset.py --train train.txt --val val.txt

# 自动扫描并划分 (当不提供划分文件时)
python prepare_fred_dataset.py --samples 1000
```

### 生成的数据集结构

```
/mnt/data/datasets/fred_drone/
├── images/
│   ├── train/   # 训练图片 (符号链接)
│   ├── val/     # 验证图片
│   └── test/    # 测试图片
├── labels/
│   ├── train/   # YOLO格式标注
│   ├── val/
│   └── test/
└── fred_drone.yaml  # 数据集配置
```

## 模型训练

### 训练脚本

```bash
python train_drone.py
```

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| epochs | 15 | 训练轮数 |
| imgsz | 640 | 输入图像尺寸 |
| batch | 16 | 批次大小 |
| device | 0 | GPU 设备 ID |
| pretrained | False | 是否使用预训练权重 |

### 训练输出

```
runs/detect/drone_yolo12n/
├── weights/
│   ├── best.pt    # 最佳模型
│   └── last.pt    # 最后模型
├── results.png    # 训练曲线
└── labels.jpg     # 标签分布
```

## 模型测试

```bash
# 测试集预测
yolo predict model=runs/detect/drone_yolo12n/weights/best.pt source=/mnt/data/datasets/fred_drone/images/test

# 单张图片
yolo predict model=runs/detect/drone_yolo12n/weights/best.pt source=image.jpg

# 视频
yolo predict model=runs/detect/drone_yolo12n/weights/best.pt source=video.mp4
```

## 模型验证

```bash
yolo val model=runs/detect/drone_yolo12n/weights/best.pt data=/mnt/data/datasets/fred_drone/fred_drone.yaml
```

## 模型导出

```bash
# ONNX
yolo export model=runs/detect/drone_yolo12n/weights/best.pt format=onnx

# TensorRT
yolo export model=runs/detect/drone_yolo12n/weights/best.pt format=engine
```

## 文件结构

```
/mnt/data/code/ultralytics/
├── prepare_fred_dataset.py   # 数据准备脚本
├── train_drone.py            # 训练脚本
├── train.txt                 # 训练集划分
├── val.txt                   # 验证集划分
├── test.txt                  # 测试集划分
└── runs/
    └── drone_yolo12n/        # 训练输出
        └── weights/
            ├── best.pt
            └── last.pt
```

## 注意事项

1. **数据量**: Demo 使用少量样本，实际训练建议使用完整数据集
2. **预训练权重**: 建议使用 `pretrained=True` 加载预训练权重以获得更好效果
3. **划分一致性**: 脚本严格按照 train.txt/val.txt/test.txt 划分，确保数据不泄露
4. **标注格式**: 支持 `x1,y1,x2,y2,class` 格式的内联标注
