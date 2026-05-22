# E-Commerce Clothing Classifier Model

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify e-commerce clothing items. It is trained and evaluated on the FashionMNIST dataset.

## Features
- **Deep Learning Framework:** PyTorch
- **Dataset:** FashionMNIST (automatically downloaded to `data/`)
- **Metrics:** Calculates Accuracy, Precision, and Recall per class using `torchmetrics`.
- **Visualization:** Generates a bar graph using `matplotlib` to compare Precision and Recall for each clothing category.

## Setup Requirements

Ensure you have Python installed, then install the required dependencies:

```bash
pip install torch torchvision torchmetrics matplotlib numpy
```

*(Note: It is recommended to use a virtual environment like `.venv` to install these dependencies.)*

## Usage

Simply run the main Python script:

```bash
python model.py
```

The script will:
1. Download the FashionMNIST dataset (if not already downloaded).
2. Train the CNN model.
3. Evaluate the model on the test dataset.
4. Display a bar chart containing Precision and Recall metrics for each clothing class.
