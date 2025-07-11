# Sketch Image Classifier Fine-Tuning

This project provides scripts and utilities to fine-tune a sketch image classifier using Hugging Face Transformers and PyTorch. It is designed for binary (or multi-class) classification of images, such as distinguishing between 'sketch' and 'not_sketch' images.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
- [Outputs](#outputs)
- [Requirements](#requirements)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)

---

## Features

- Fine-tune image classification models (e.g., SigLIP) on custom sketch datasets
- Uses Hugging Face `transformers`, `datasets`, and `evaluate` libraries
- Supports GPU acceleration via PyTorch
- Modular and extensible codebase
- Example scripts for both quick-start and advanced usage

## Project Structure

```
.
├── dataset/
│   ├── train/
│   │   ├── sketch/        # Training images for 'sketch' class
│   │   └── not_sketch/    # Training images for 'not_sketch' class
│   └── val/
│       ├── sketch/        # Validation images for 'sketch' class
│       └── not_sketch/    # Validation images for 'not_sketch' class
├── finetune_sketch_classifier_og.py      # Main training script
├── finetune_sketch_classifier_template.py # Template for advanced/argparse usage
├── install.py              # Automated environment and dependency setup
├── run.py                  # Script to run training in the venv
├── requirements.txt        # Python dependencies
├── sketch-finetuned/       # Output directory for fine-tuned model and checkpoints
└── README.md               # This file
```

## Installation

**Requirements:**

- Windows OS (uses Windows-style venv activation)
- Python 3.9 (must be available as `py -3.9`)
- CUDA 12.1 (for GPU support; adjust torch install if using CPU or other CUDA version)

**Steps:**

1. Open a terminal in the project root directory.
2. Run the following command to set up the environment and install dependencies:

   ```sh
   py -3.9 install.py
   ```

   This will:

   - Create a virtual environment in `venv/` using Python 3.9
   - Install PyTorch (torch, torchvision, torchaudio) with CUDA 12.1 support
   - Install all required Python packages from `requirements.txt`

3. To activate the environment manually (if needed):
   ```sh
   venv\Scripts\activate
   ```

## Dataset Structure

Your dataset should be organized as follows:

```
dataset/
  train/
    sketch/       # e.g., 0001.jpg, 0002.png, ...
    not_sketch/   # e.g., 0001.jpg, 0002.png, ...
  val/
    sketch/       # e.g., 0001.jpg, 0002.png, ...
    not_sketch/   # e.g., 0001.jpg, 0002.png, ...
```

- Each class (e.g., `sketch`, `not_sketch`) is a subfolder containing images for that class.
- Supported image formats: `.jpg`, `.png`, etc.
- You can add more classes by creating additional subfolders.

## Usage

### Quick Start (Recommended)

After installation, run the training script using:

```sh
py run.py
```

This will:

- Use the virtual environment's Python
- Run `finetune_sketch_classifier_og.py` to start training

### Manual Run

Alternatively, you can activate the environment and run the script manually:

```sh
venv\Scripts\activate
python finetune_sketch_classifier_og.py
```

### Custom Training (Advanced)

For more flexibility (e.g., changing model, hyperparameters, or dataset location), use the template script:

```sh
python finetune_sketch_classifier_template.py --data_dir ./dataset --output_dir ./sketch-finetuned --model_name <pretrained-model-or-path> --epochs 4 --batch_size 8 --lr 4e-5 --image_size 224
```

See the script for all available arguments.

## Outputs

- The fine-tuned model and processor are saved to `sketch-finetuned/` by default.
- Checkpoints are saved during training (e.g., `sketch-finetuned/checkpoint-*/`)
- Training logs are saved in `sketch-finetuned/logs/`
- The final model and processor can be loaded for inference using Hugging Face Transformers.

## Requirements

- Python 3.9
- torch, torchvision, torchaudio (CUDA 12.1)
- transformers >= 4.20.0
- datasets >= 2.0.0
- evaluate
- numpy < 2
- scikit-learn
- setuptools
- certify

All dependencies are installed automatically by `install.py`.

## Customization

- To add new classes, simply add new subfolders under `train/` and `val/`.
- To use a different model, change the `MODEL_NAME` in the script or use the template with `--model_name`.
- Hyperparameters (epochs, batch size, learning rate, image size) can be changed in the script or via command-line arguments in the template.

## Troubleshooting

- **Python 3.9 not found:** Ensure Python 3.9 is installed and available as `py -3.9` on your system.
- **CUDA version mismatch:** If you do not have CUDA 12.1, edit `install.py` to use the correct torch index URL, or install the CPU version.
- **Missing dependencies:** Re-run `install.py` to reinstall all requirements.
- **Dataset errors:** Ensure your dataset follows the required folder structure and contains valid images.

---

For further customization, see the comments and docstrings in `finetune_sketch_classifier_template.py`.
