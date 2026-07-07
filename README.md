# Land Use Classification

A deep learning pipeline that classifies satellite imagery into land use categories using a custom Convolutional Neural Network (CNN) built in PyTorch. Applications include urban planning, environmental monitoring, and geographic information systems (GIS).

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
