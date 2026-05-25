#!/usr/bin/env python
# coding: utf-8

"""
Brain Tumour MRI Classification with CNNs and Transfer Learning
===============================================================

Author: Newton Fernihough

A deep-learning pipeline for multi-class classification of brain MRI
scans (meningioma, glioma, pituitary) using the public Brain Tumor
dataset distributed as MATLAB ``.mat`` files inside four zipped archives.
The script handles dataset extraction, fold-based cross-validation
splitting, class-imbalance treatment, training of several architectures,
and a side-by-side comparison of their performance.

Models compared:
    - SimpleCNN  (a small custom convolutional baseline)
    - DeeperCNN  (a deeper custom convolutional network)
    - ResNet18   (transfer learning from ImageNet)
    - EfficientNetB0 (transfer learning from ImageNet)

Pipeline features:
    - Five-fold cross-validation following the provided cvind.mat splits
    - Optional tumour-mask cropping, class weighting, and weighted sampling
    - Early stopping, learning-rate scheduling, and checkpointing
    - Per-fold and aggregated metrics, confusion matrices, and plots

Data:
    The four ``brainTumorDataPublic_*.zip`` archives plus ``cvind.mat``
    are expected in a ``dataset/`` folder alongside the script.

AI-assistance disclosure
------------------------
ChatGPT was used to assist with specific portions of this work, namely:
    - parts of the MATLAB / HDF5 dataset-loading utilities
    - weighted sampling setup for class imbalance
    - parts of the training pipeline (scheduler, checkpoint, early stopping)
    - structuring the experiment-running and results-saving workflow
    - general code review, polishing and cleanup

The model choices, preprocessing decisions, experiment settings, result
comparisons, and interpretation of the final results were my own.
"""

# **Imports**

from pathlib import Path
import json
import zipfile
import random

import numpy as np
import pandas as pd
import h5py
import scipy.io as sio
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models


# **User settings**

# In[76]:


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_ROOT = PROJECT_ROOT / "dataset"

# The four zip files that make up the public brain tumour dataset.
ZIP_FILES = [
    DATA_ROOT / "brainTumorDataPublic_1-766.zip",
    DATA_ROOT / "brainTumorDataPublic_767-1532.zip",
    DATA_ROOT / "brainTumorDataPublic_1533-2298.zip",
    DATA_ROOT / "brainTumorDataPublic_2299-3064.zip",
]

# Fold index file and extraction destination.
CVIND_PATH = DATA_ROOT / "cvind.mat"
EXTRACT_DIR = DATA_ROOT / "brain_tumor_mat_files"

# Folder for outputs
OUTPUT_DIR = PROJECT_ROOT / "brain_tumour_outputs_current_test"

# Marker note:
# To test the code more quickly, the following settings can be reduced, for example:
# FOLDS_TO_RUN = [1] or [1, 2]
# MODEL_NAMES = ["SimpleCNN", "DeeperCNN"]
# NUM_EPOCHS = a smaller value such as 3 or 5

# Main experiment settings
FOLDS_TO_RUN = [1, 2, 3, 4, 5]
MODEL_NAMES = ["SimpleCNN", "DeeperCNN", "ResNet18", "EfficientNetB0"]

NUM_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
PATIENCE = 5
IMAGE_SIZE = 224
NUM_WORKERS = 0
SEED = 42

# Settings for controlled comparisons
USE_MASK_CROP = False
USE_CLASS_WEIGHTS = False
USE_WEIGHTED_SAMPLER = False

# Class labels
CLASS_NAMES = {1: "meningioma", 2: "glioma", 3: "pituitary"}
CLASS_INDEX_TO_NAME = {0: "meningioma", 1: "glioma", 2: "pituitary"}
NUM_CLASSES = 3

# Run on CPU
DEVICE = torch.device("cpu")


# **Reproducibility**

# In[78]:


def set_seed(seed: int = 42):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)


# **File extraction and setup**

# In[80]:


def ensure_directories(output_dir):
    """Create output folders."""
    output_dir.mkdir(parents=True, exist_ok=True)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

def extract_zip_files():
    """Extract the dataset zip files if needed."""
    existing_mats = [p for p in EXTRACT_DIR.rglob("*.mat") if p.name.lower() != "cvind.mat"]
    
    # Count extracted MATLAB files to avoid re-extracting the dataset unnecessarily.
    if len(existing_mats) > 100:
        print(f"Found {len(existing_mats)} extracted .mat files already. Skipping extraction.")
        return

    print("Extracting dataset zip files...")

    for zip_path in ZIP_FILES:
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing zip file: {zip_path}")

        print(f"  Extracting: {zip_path.name}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(EXTRACT_DIR)

    total_mats = len([p for p in EXTRACT_DIR.rglob("*.mat") if p.name.lower() != "cvind.mat"])
    print(f"Extraction complete. Found {total_mats} image .mat files.")


# **MATLAB data loading**

# In[82]:


""" AI-assissted section: ChatGPT helped with parts of this MATLAB/.mat loading 
pipeline, especially the fallback between scipy.io and h5py and the extraction 
of image, label,PID, and tumour-mask data.
The fallback was included as a robustness measure, even though the
dataset may not require both loading routes in practice.
I used this section to return the file contents in a consistent 
structure for later preprocessing and dataframe building."""


def _decode_pid_from_mat(pid_dataset):
    """Decode PID data from a MATLAB array."""
    arr = np.array(pid_dataset).squeeze()

    if arr.size == 0:
        return ""

    if np.issubdtype(arr.dtype, np.integer):
        try:
            return "".join(chr(int(x)) for x in arr.tolist())
        except Exception:
            return str(arr.tolist())

    return str(arr.tolist())

def _safe_read_hdf5_mat(mat_path):
    """Read one HDF5 MATLAB file."""
    with h5py.File(mat_path, "r") as f:
        if "cjdata" not in f:
            raise KeyError(f"'cjdata' group not found in {mat_path}")

        cjdata = f["cjdata"]

        image = np.array(cjdata["image"])

        # MATLAB arrays sometimes load transposed, so fix the orientation here
        if image.ndim >= 2:
            image = image.T
        image = np.array(image)

        label = int(np.array(cjdata["label"]).squeeze())

        pid = ""
        if "PID" in cjdata:
            pid = _decode_pid_from_mat(cjdata["PID"])

        tumor_mask = None
        if "tumorMask" in cjdata:
            tumor_mask = np.array(cjdata["tumorMask"])
            if tumor_mask.ndim >= 2:
                tumor_mask = tumor_mask.T
            tumor_mask = np.array(tumor_mask)

    return {
        "image": image,
        "label": label,
        "pid": pid,
        "tumorMask": tumor_mask,
    }

def read_mat_file(mat_path):
    """Read one image .mat file."""
    return _safe_read_hdf5_mat(mat_path)

def load_cvind(cvind_path):
    """Load fold assignments from cvind.mat."""
    
    # Try scipy first
    try:
        mat = sio.loadmat(cvind_path)
        keys = [k for k in mat.keys() if not k.startswith("__")]

        if not keys:
            raise ValueError("No usable variable found in cvind.mat")

        arr = mat["cvind"] if "cvind" in mat else mat[keys[0]]
        arr = np.array(arr).squeeze().astype(int)
        return arr

    # If that fails, try h5py instead
    except (NotImplementedError, ValueError):
        with h5py.File(cvind_path, "r") as f:
            keys = list(f.keys())

            if not keys:
                raise ValueError("No usable variable found in cvind.mat")

            arr = np.array(f["cvind"]) if "cvind" in f else np.array(f[keys[0]])
            arr = np.array(arr).squeeze()

            if arr.ndim > 1:
                arr = arr.reshape(-1)

            arr = arr.astype(int)
            return arr

def build_dataframe():
    """Build the main dataframe."""
    mat_files = sorted(
        [p for p in EXTRACT_DIR.rglob("*.mat") if p.name.lower() != "cvind.mat"],
        key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or "0")
    )

    if not mat_files:
        raise FileNotFoundError("No .mat files found after extraction.")

    cvind = load_cvind(CVIND_PATH)

    if len(cvind) < len(mat_files):
        raise ValueError(
            f"cvind length ({len(cvind)}) is smaller than number of .mat files ({len(mat_files)})."
        )

    records = []
    for idx, mat_path in enumerate(mat_files):
        item = read_mat_file(mat_path)
        label_raw = int(item["label"])

        if label_raw not in CLASS_NAMES:
            raise ValueError(f"Unexpected label {label_raw} in {mat_path}")

        records.append({
            "file_path": str(mat_path),
            "label_raw": label_raw,
            "label_idx": label_raw - 1,
            "label_name": CLASS_NAMES[label_raw],
            "pid": item["pid"],
            "fold": int(cvind[idx]),
        })

    return pd.DataFrame(records)


# **Preprocessing**

# In[84]:


def normalize_to_uint8(image):
    """Normalise an image to uint8."""
    image = image.astype(np.float32)

    min_val = image.min()
    max_val = image.max()

    if max_val - min_val < 1e-8:
        return np.zeros_like(image, dtype=np.uint8)

    image = (image - min_val) / (max_val - min_val)
    image = (image * 255).clip(0, 255).astype(np.uint8)
    return image

def crop_from_mask(image, mask, pad=8):
    """Crop an image using the tumour mask."""
    if mask is None:
        return image

    mask = np.array(mask)

    if mask.ndim != 2:
        return image

    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return image

    x1 = max(int(xs.min()) - pad, 0)
    x2 = min(int(xs.max()) + pad, image.shape[1] - 1)
    y1 = max(int(ys.min()) - pad, 0)
    y2 = min(int(ys.max()) + pad, image.shape[0] - 1)

    cropped = image[y1:y2 + 1, x1:x2 + 1]

    if cropped.size == 0:
        return image

    return cropped

class BrainTumorDataset(Dataset):
    """Dataset for brain tumour MRI images."""
    def __init__(self, dataframe, transform=None, use_mask_crop=True):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.use_mask_crop = use_mask_crop

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = read_mat_file(row["file_path"])

        image = item["image"]
        mask = item["tumorMask"]

        if self.use_mask_crop:
            image = crop_from_mask(image, mask)

        image = normalize_to_uint8(image)
        
        # Convert the grayscale MRI slice to RGB so it can be passed into pretrained torchvision models.
        pil_img = Image.fromarray(image).convert("RGB")

        if self.transform is not None:
            image_tensor = self.transform(pil_img)
        else:
            image_tensor = transforms.ToTensor()(pil_img)

        label = int(row["label_idx"])
        return image_tensor, label

def get_transforms(image_size=224):
    """Create training and validation transforms."""
    
    # Training transform: includes augmentation to improve robustness.
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.15),
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=5,
        ),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    # Validation transform: no augmentation, only resizing and normalisation.
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, eval_transform


# **Models**

# In[86]:


class SimpleCNN(nn.Module):
    """Simple custom CNN baseline."""
    def __init__(self, num_classes=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """Run a forward pass."""
        return self.classifier(self.features(x))


class DeeperCNN(nn.Module):
    """Deeper custom CNN baseline."""
    def __init__(self, num_classes=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """Run a forward pass."""
        return self.classifier(self.features(x))


def build_resnet18(num_classes=3):
    """Build ResNet18 for three classes."""
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(weights=None)

    # Replace the final fully connected layer so ResNet18 outputs the three tumour classes.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes=3):
    """Build EfficientNetB0 for three classes."""
    try:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
    except Exception:
        model = models.efficientnet_b0(weights=None)

    # Replace the EfficientNet classifier head for three-class classification.
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def get_model_dict(model_names=None):
    """Return the selected models."""
    all_models = {
        "SimpleCNN": SimpleCNN(NUM_CLASSES),
        "DeeperCNN": DeeperCNN(NUM_CLASSES),
        "ResNet18": build_resnet18(NUM_CLASSES),
        "EfficientNetB0": build_efficientnet_b0(NUM_CLASSES),
    }

    if model_names is None:
        return all_models

    return {name: all_models[name] for name in model_names}


# **Training helpers**

# In[88]:


def compute_class_weights(labels):
    """Compute class weights."""
    labels = np.array(labels)
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    total = counts.sum()

    # Smaller classes get larger weights
    weights = total / (NUM_CLASSES * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


"""AI-assissted section: ChatGPT helped with the weighted sampling setup below, 
especially the construction of per-sample weights for WeightedRandomSampler.
Samples from rarer classes are given higher weights based on the inverse 
of class frequency, so those classes appear more often in training batches.
I used this to test whether oversampling improved class balance during training."""


def build_sampler(labels):
    """Build a weighted sampler."""
    labels = np.array(labels)
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_weights = 1.0 / np.maximum(class_counts, 1)

    # Give each sample a weight based on its class
    sample_weights = class_weights[labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def train_one_epoch(model, loader, criterion, optimizer):
    """Train for one epoch."""
    model.train()

    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagation and optimiser step
        loss.backward()
        optimizer.step()

        # Store loss and predictions
        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Mean loss and accuracy for the epoch
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Evaluate on the validation set."""
    model.eval()

    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Macro-averaged metrics are used so each tumour class contributes equally to the summary score
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    return {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "labels": all_labels,
        "preds": all_preds,
    }


def save_history_plot(history, save_path, title):
    """Save training history plots."""
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} - Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    # Accuracy plot
    acc_path = str(save_path).replace("_loss.png", "_accuracy.png")
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title} - Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path, dpi=200)
    plt.close()


def save_confusion_matrix(cm, labels, save_path, title):
    """Save a confusion matrix plot."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    # Write the values inside the matrix
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


"""AI-assissted section: ChatGPT helped with parts of the training wrapper below,
especially the scheduler, checkpoint saving, and early-stopping logic.
These functions train the model epoch by epoch, tracks validation 
performance, reduces the learning rate if validation loss stalls,
saves the best checkpoint, and stops early when improvement stops.
I still chose the settings used in the experiments and used this function
to run the final training and validation comparisons reported in the assignment."""


def train_model(
    model_name,
    model,
    train_loader,
    val_loader,
    class_weights,
    model_dir,
    num_epochs,
    use_class_weights=True,
):
    """Train one model and save the best checkpoint."""
    
    # Make sure the output folder exists
    model_dir.mkdir(parents=True, exist_ok=True)

    # Move model to the selected device
    model = model.to(DEVICE)

    # Use weighted loss if enabled
    if use_class_weights:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Reduce learning rate if validation loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    checkpoint_path = model_dir / f"{model_name}_best.pt"

    # Store training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1_macro": [],
    }

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_metrics = evaluate(model, val_loader, criterion)

        # Save metrics for this epoch
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])

        scheduler.step(val_metrics["loss"])

        print(
            f"{model_name} | Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1_macro']:.4f}"
        )

        # Save the best checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1

        # Stop early if validation loss has not improved
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping triggered for {model_name} at epoch {epoch + 1}.")
            break

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint was not saved for {model_name}: {checkpoint_path}")

    # Reload the best checkpoint before final evaluation
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    final_val_metrics = evaluate(model, val_loader, criterion)

    return model, history, final_val_metrics, best_epoch, checkpoint_path


# **Cross-validation / Running the experiment**

# In[ ]:


def run_experiment(
    output_dir,
    folds_to_run,
    num_epochs,
    use_mask_crop,
    model_names,
    use_class_weights,
    use_weighted_sampler,
):
    """Run the full experiment."""

    
    """AI-assissted section: ChatGPT helped structure parts of this experiment-running 
    section, particularly the repeated fold handling, automated saving of results,
    and export of summaries such as JSON histories and CSV performance tables.
    These functions rebuild the training and validation splits for each
    selected fold, applies the chosen settings, trains each requested model,
    and saves the outputs in a consistent format for later comparison.
    The experiment design, selected settings, model comparisons, and interpretation
    of the final results were decided by me."""
    
    
    # Prepare folders and extract the dataset if needed
    ensure_directories(output_dir)
    extract_zip_files()

    # Build the main metadata table
    df = build_dataframe()

    # Print a quick dataset summary
    print("\nDataset summary:")
    print(df["label_name"].value_counts())
    print("\nFold counts:")
    print(df["fold"].value_counts().sort_index())

    # Save the dataframe used for the experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "dataset_index.csv", index=False)

    # Build train and validation transforms
    train_transform, eval_transform = get_transforms(IMAGE_SIZE)

    # Keep only folds that actually exist
    available_folds = sorted(df["fold"].unique().tolist())
    unique_folds = [f for f in folds_to_run if f in available_folds]

    if not unique_folds:
        raise ValueError(f"No valid folds found. Available folds: {available_folds}")

    print(f"\nUsing folds: {unique_folds}")
    print(f"Mask crop enabled: {use_mask_crop}")
    print(f"Class weights enabled: {use_class_weights}")
    print(f"Weighted sampler enabled: {use_weighted_sampler}")
    print(f"Models: {model_names}")
    print(f"Epochs: {num_epochs}")

    all_results = []

    # Loop through each validation fold
    for fold_id in unique_folds:
        print("\n" + "=" * 80)
        print(f"STARTING FOLD {fold_id}")
        print("=" * 80)

        fold_output_dir = output_dir / f"fold_{fold_id}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        # Split into training and validation data
        train_df = df[df["fold"] != fold_id].reset_index(drop=True)
        val_df = df[df["fold"] == fold_id].reset_index(drop=True)

        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError(f"Fold {fold_id} produced an empty train or validation split.")

        # Compute class weights from the training data
        class_weights = compute_class_weights(train_df["label_idx"].values)

        # Build sampler if enabled
        sampler = None
        if use_weighted_sampler:
            sampler = build_sampler(train_df["label_idx"].values)

        # Build datasets
        train_dataset = BrainTumorDataset(
            train_df, transform=train_transform, use_mask_crop=use_mask_crop
        )
        val_dataset = BrainTumorDataset(
            val_df, transform=eval_transform, use_mask_crop=use_mask_crop
        )

        # Build dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            shuffle=False if sampler is not None else True,
            num_workers=NUM_WORKERS,
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=False,
        )

        # Get the selected models
        models_dict = get_model_dict(model_names=model_names)

        # Train each model on this fold
        for model_name, model in models_dict.items():
            print("\n" + "-" * 60)
            print(f"Training model: {model_name} | Fold: {fold_id}")
            print("-" * 60)

            model_dir = fold_output_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            _, history, metrics, best_epoch, checkpoint_path = train_model(
                model_name=model_name,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=class_weights,
                model_dir=model_dir,
                num_epochs=num_epochs,
                use_class_weights=use_class_weights,
            )

            # Save training history
            with open(model_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

            # Save the training history plot
            save_history_plot(
                history,
                model_dir / f"{model_name}_loss.png",
                title=f"{model_name} Fold {fold_id}"
            )

            # Save confusion matrix
            cm = confusion_matrix(metrics["labels"], metrics["preds"])
            save_confusion_matrix(
                cm,
                labels=[CLASS_INDEX_TO_NAME[i] for i in range(NUM_CLASSES)],
                save_path=model_dir / f"{model_name}_confusion_matrix.png",
                title=f"{model_name} Fold {fold_id}"
            )

            # Save classification report
            report_dict = classification_report(
                metrics["labels"],
                metrics["preds"],
                target_names=[CLASS_INDEX_TO_NAME[i] for i in range(NUM_CLASSES)],
                output_dict=True,
                zero_division=0,
            )

            with open(model_dir / "classification_report.json", "w") as f:
                json.dump(report_dict, f, indent=2)

            # Save summary results for this fold
            result_row = {
                "fold": fold_id,
                "model": model_name,
                "best_epoch": best_epoch,
                "val_accuracy": metrics["accuracy"],
                "val_precision_macro": metrics["precision_macro"],
                "val_recall_macro": metrics["recall_macro"],
                "val_f1_macro": metrics["f1_macro"],
                "checkpoint_path": str(checkpoint_path),
                "mask_crop": use_mask_crop,
                "class_weights": use_class_weights,
                "weighted_sampler": use_weighted_sampler,
            }
            all_results.append(result_row)

            print(f"Finished {model_name} on fold {fold_id}.")
            print(result_row)

    # Save full results table
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "model_comparison_results_by_fold.csv", index=False)

    # Save grouped mean/std summary
    avg_df = (
        results_df.groupby("model")[["val_accuracy", "val_precision_macro", "val_recall_macro", "val_f1_macro"]]
        .agg(["mean", "std"])
        .round(4)
    )
    avg_df.to_csv(output_dir / "model_comparison_results_average.csv")

    # Save a flatter summary table as well
    flat_rows = []
    for model_name, group in results_df.groupby("model"):
        flat_rows.append({
            "model": model_name,
            "accuracy_mean": group["val_accuracy"].mean(),
            "accuracy_std": group["val_accuracy"].std(ddof=0),
            "precision_macro_mean": group["val_precision_macro"].mean(),
            "recall_macro_mean": group["val_recall_macro"].mean(),
            "f1_macro_mean": group["val_f1_macro"].mean(),
        })

    pd.DataFrame(flat_rows).round(4).to_csv(
        output_dir / "model_comparison_results_average_flat.csv", index=False
    )

    # Save the settings used
    settings_summary = {
        "folds_to_run": folds_to_run,
        "num_epochs": num_epochs,
        "use_mask_crop": use_mask_crop,
        "use_class_weights": use_class_weights,
        "use_weighted_sampler": use_weighted_sampler,
        "model_names": model_names,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "patience": PATIENCE,
        "image_size": IMAGE_SIZE,
    }

    with open(output_dir / "experiment_settings.json", "w") as f:
        json.dump(settings_summary, f, indent=2)

    print("\nAll experiments complete.")
    print(f"Results saved to: {output_dir}")


# Run the experiment
run_experiment(
    output_dir=OUTPUT_DIR,
    folds_to_run=FOLDS_TO_RUN,
    num_epochs=NUM_EPOCHS,
    use_mask_crop=USE_MASK_CROP,
    model_names=MODEL_NAMES,
    use_class_weights=USE_CLASS_WEIGHTS,
    use_weighted_sampler=USE_WEIGHTED_SAMPLER,
)

