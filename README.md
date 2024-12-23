# Vision Transformer (ViT) Implementation in PyTorch

This repository provides a comprehensive implementation of the Vision Transformer (ViT) model in PyTorch. ViT leverages transformer architectures, originally designed for natural language processing, to achieve state-of-the-art performance in image classification tasks.

## Repository Structure

The project is organized as follows:


- **`models/`**: Contains the core components of the ViT model.
    - `__init__.py`: Initializes the models package.
    - `ViT.py`: Defines the Vision Transformer class.
    - `Transfomer.py`: Implements the Transformer Encoder block.
    - `MLPBlock.py`: Contains the Multi-Layer Perceptron (MLP) block used in the transformer.
    - `PatchEmbedding.py`: Handles the patch embedding layer.
    - `MultiHeadAttention.py`: Implements the multi-head attention mechanism.

- **`train.py`**: Script to train the ViT model on the CIFAR-100 dataset.
- **`utils.py`**: Includes utility functions for data loading and preprocessing.
- **`README.md`**: This file, providing an overview of the project.

## Requirements

Ensure you have the following Python packages installed:

- `torch`
- `torchvision`
- `tqdm`
- `numpy`
- `matplotlib`

You can install them using pip:

```bash
pip install torch torchvision numpy matplotlib tqdm 
```

```bash
python -m venv vit
source vit/bin/activate
pip install -r requirements.txt
python train.py
```
