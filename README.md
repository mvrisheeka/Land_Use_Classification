# Land Use Classification

A deep learning pipeline that classifies satellite imagery into land use categories using a custom Convolutional Neural Network (CNN) built in PyTorch. Applications include urban planning, environmental monitoring, and geographic information systems (GIS).

## Overview

This project trains a CNN to recognize land use types from aerial/satellite photographs. It covers the full workflow: downloading the dataset, exploratory data analysis (EDA), preprocessing, model training, and evaluation.

## Dataset

**[UC Merced Land Use Dataset](http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip)**

| Property | Value |
|---|---|
| Total images | 2,100 |
| Classes | 21 land use categories |
| Images per class | 100 |
| Image size | 256×256 pixels |
| Format | TIFF, RGB, high-resolution overhead imagery |

**Categories:** agricultural, airplane, baseball diamond, beach, buildings, chaparral, dense residential, forest, freeway, golf course, harbor, intersection, medium residential, mobile home park, overpass, parking lot, river, runway, sparse residential, storage tanks, tennis court

The notebook automatically downloads and extracts the dataset on first run.

## Pipeline

1. **Setup** – Install/import required libraries
2. **Dataset** – Download and extract the UC Merced dataset
3. **Exploratory Data Analysis** – Class distribution, sample images per class, image property and color-channel analysis
4. **Data Preprocessing** – Custom `Dataset`/`DataLoader` pipeline, train/val/test split (70/15/15), data augmentation (random flips, rotation, resized crop, color jitter), normalization (ImageNet statistics)
5. **Model Training and Comparison** – Custom lightweight CNN, trained with class-weighted loss and a cosine annealing LR schedule
6. **Evaluation** – Classification report, confusion matrix, and visual inspection of correct/incorrect predictions

## Model Architecture

A lightweight custom CNN (not a pretrained backbone):

- 4 convolutional blocks (32 → 64 → 128 → 256 channels), each with BatchNorm + ReLU, the first three followed by max pooling
- Global average pooling → Flatten
- Fully connected head: 256 → 128 (ReLU, Dropout 0.6) → `num_classes`

## Training Configuration

| Setting | Value |
|---|---|
| Batch size | 32 |
| Optimizer | AdamW (lr = 3e-4, weight decay = 1e-4) |
| LR schedule | Cosine annealing |
| Loss | Cross-entropy with class weighting |
| Epochs | 15 |
| Gradient clipping | Max norm 1.0 |

The number of classes used for training is configurable (`NUM_CLASSES_TO_USE` — set to `None` to use all 21 classes, or a smaller number for faster experimentation).

## Requirements

```
torch
torchvision
matplotlib
seaborn
scikit-learn
pandas
numpy
pillow
requests
tqdm
```

These are installed automatically in the notebook's setup cell.

## Usage

1. Open `Land_Use_Classification.ipynb` in Jupyter.
2. Run all cells in order — the dataset will be downloaded automatically to `/home/land_use_data`.
3. Adjust `NUM_CLASSES_TO_USE` and training hyperparameters (batch size, learning rate, epochs) as needed.
4. The best model checkpoint is saved to `/home/best_model.pth` and reloaded automatically for test-set evaluation.

## Outputs

- Training/validation loss and accuracy curves
- Classification report (precision, recall, F1 per class)
- Confusion matrix heatmap
- Grid of correctly/incorrectly classified test images

## Notes

- A GPU is recommended for training but not required.
- Class weighting is used in the loss function to account for class imbalance.
- Data augmentation is applied only to the training set; validation/test images use resize + normalize only.
