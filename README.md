# U-Net Semantic Segmentation — Oxford-IIIT Pet Dataset

A PyTorch implementation of U-Net for binary semantic segmentation, trained on the Oxford-IIIT Pet Dataset. This project was developed as part of the **AI and Deep Learning** course (Academic Year 2024–2025).

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Pretrained Model](#pretrained-model)

---

## Overview

Semantic segmentation is the task of assigning a class label to every pixel in an image. This project implements a simplified **U-Net** encoder–decoder architecture for binary segmentation — separating the pet (foreground) from the background in pet images.

---

## Dataset

**Oxford-IIIT Pet Dataset**
- 7,390 images of cats and dogs across 37 breeds
- Binary segmentation masks: pet (foreground) vs background
- Split: 90% train / 10% val from `trainval.txt`, official `test.txt` for testing
- Dataset is automatically downloaded via `torchvision`

| Split | Size |
|---|---|
| Train | ~3,312 images |
| Val | ~368 images |
| Test | ~3,669 images |

---

## Model Architecture

Simplified U-Net with encoder–decoder structure and skip connections.

```
Input [3, 256, 256]
    ↓
Encoder Block 1 (64)   → skip connection 1
Encoder Block 2 (128)  → skip connection 2
Encoder Block 3 (256)  → skip connection 3
    ↓
Bottleneck (512)
    ↓
Decoder Block 1 (256)  ← concat skip 3
Decoder Block 2 (128)  ← concat skip 2
Decoder Block 3 (64)   ← concat skip 1
    ↓
Output Conv (1×1) + Sigmoid
Output [1, 256, 256]
```

**Key components:**
- Each encoder/decoder block uses two 3×3 convolutions with BatchNorm and ReLU
- Downsampling via 2×2 MaxPooling
- Upsampling via Transposed Convolution
- Skip connections concatenate encoder and decoder feature maps
- Sigmoid activation on output for binary segmentation
- Total parameters: ~7.7 million

---

## Results

Training for 25 epochs with Adam optimizer (lr=1e-3) and combined BCE + Dice loss.

### Validation Metrics (Best Model — Epoch 25)

| Metric | Score |
|---|---|
| Pixel Accuracy | 90.88% |
| IoU — Background | 0.8739 |
| IoU — Foreground | 0.7437 |
| Mean IoU (mIoU) | 0.8088 |
| Dice Coefficient | 0.8500 |

### Training Progress

| Epoch | Val Loss | Val Acc | Val mIoU | Val Dice |
|---|---|---|---|---|
| 1 | 0.5479 | 66.85% | 0.4941 | 0.6009 |
| 5 | 0.3478 | 84.43% | 0.6916 | 0.7349 |
| 10 | 0.2820 | 88.32% | 0.7520 | 0.7888 |
| 17 | 0.2233 | 90.40% | 0.7981 | 0.8390 |
| 25 | 0.2132 | 90.88% | 0.8088 | 0.8500 |

---

## Project Structure

```
├── evaluate.py         # Evaluation, visualisation, training curves
├── Image_Segmentation.ipynb      # Full Jupyter/Colab notebook with data loading, and preprocessing, with model training
├── predictions.png     # Sample predictions visualisation
├── training_curves.png # Loss, accuracy, IoU, Dice curves
└── README.md
```

---

## Installation

**Requirements:**
```bash
pip install torch torchvision matplotlib numpy Pillow torchsummary
```

**Clone the repository:**
```bash
git clone https://github.com/your_username/unet-segmentation.git
cd unet-segmentation
```

---

## Usage

### 1. Train the model
```python
from dataset import get_dataloaders
from model import UNet
from train import train

# Load data (downloads automatically on first run)
train_loader, val_loader, test_loader = get_dataloaders(
    root="./data", img_size=256, batch_size=8
)

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = UNet(in_channels=3, num_classes=1).to(device)

# Train

history = train(
    model        = model,
    train_loader = train_loader,
    val_loader   = val_loader,
    num_epochs   = 25,
    lr           = 1e-3,
    device       = device,
    save_path    = "best_model.pth",
)
```

### 2. Evaluate the model
```python
from evaluate import load_best_model, evaluate_model, print_results
from evaluate import visualize_predictions, plot_history

model   = load_best_model(model, save_path="best_model.pth", device=device)
results = evaluate_model(model, test_loader, device=device)
print_results(results)
visualize_predictions(model, test_loader, device=device, num_samples=6)
plot_history(history)
```

---

## Pretrained Model

The pretrained model weights are available on Google Drive:

📥 **[Download best_model.pth](https://drive.google.com/file/d/1zskmQQ0XDU8OutrTQjQjlmrQ_kvFb8rM/view?usp=drive_link)**

Place the downloaded file in the root directory of the project.

---

## Training Details

| Setting | Value |
|---|---|
| Framework | PyTorch |
| Image size | 256 × 256 |
| Batch size | 8 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Loss function | BCE + Dice (50/50) |
| Scheduler | ReduceLROnPlateau |
| Early stopping patience | 7 epochs |
| Hardware | NVIDIA T4 GPU (Google Colab) |

---

## Acknowledgements

- U-Net architecture: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- Dataset: [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- Course: AI and Deep Learning — Academic Year 2024–2025
