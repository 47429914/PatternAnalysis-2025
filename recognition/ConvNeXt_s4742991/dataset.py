import os
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from sklearn.model_selection import train_test_split
import numpy as np

# Constants
DATA_DIR = "/home/groups/comp3710/ADNI/AD_NC"
IMG_SIZE = 224
VAL_SPLIT = 0.15

# Custom Dataset
class ADNIDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.transform = transform
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, sample["label"], sample["patient_id"]

# Compute mean and std from training set
def compute_mean_std(dataset):
    if len(dataset) == 0:
        print("⚠️ Warning: Training dataset is empty. Using default mean=0.5, std=0.5")
        return 0.5, 0.5  # Fallback values to avoid division by zero
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=1)
    mean = 0.0
    std = 0.0
    total_images = 0
    print("🔄 Computing mean and std from training set...")
    for images, _, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, -1)
        mean += images.mean(dim=1).sum()
        std += images.std(dim=1).sum()
        total_images += batch_samples
    if total_images == 0:
        print("⚠️ Warning: No images processed in DataLoader. Using default mean=0.5, std=0.5")
        return 0.5, 0.5
    mean /= total_images
    std /= total_images
    print(f"✅ Computed mean: {mean.item()}")
    print(f"✅ Computed std: {std.item()}")
    return mean.item(), std.item()

# Data loader factory with train-val-test split
def get_dataloaders(batch_size=32, num_workers=1):
    # Load all training samples
    train_samples = []
    train_split_dir = os.path.join(DATA_DIR, "train")
    print(f"🔍 Checking train directory: {train_split_dir}")
    if not os.path.exists(train_split_dir):
        raise FileNotFoundError(f"Train directory {train_split_dir} does not exist")
    
    for label_name in ["AD", "NC"]:
        class_dir = os.path.join(train_split_dir, label_name)
        print(f"🔍 Checking class directory: {class_dir}")
        if not os.path.exists(class_dir):
            print(f"⚠️ Warning: Class directory {class_dir} does not exist")
            continue
        for fname in os.listdir(class_dir):
            if fname.endswith(".jpeg"):
                try:
                    patient_id, scan_num = fname.replace(".jpeg", "").split("_")
                    path = os.path.join(class_dir, fname)
                    train_samples.append({
                        "path": path,
                        "label": 1 if label_name == "AD" else 0,
                        "patient_id": int(patient_id),
                        "scan_num": int(scan_num)
                    })
                except ValueError as e:
                    print(f"⚠️ Warning: Skipping file {fname} due to invalid format: {e}")
    
    print(f"🔢 Total training samples found: {len(train_samples)}")
    if not train_samples:
        raise ValueError("No valid training samples found. Check data directory and file formats.")

    # Group samples by patient ID
    patient_to_samples = {}
    for sample in train_samples:
        pid = sample["patient_id"]
        if pid not in patient_to_samples:
            patient_to_samples[pid] = []
        patient_to_samples[pid].append(sample)

    # Get patient IDs and their labels
    patient_ids = list(patient_to_samples.keys())
    patient_labels = [patient_to_samples[pid][0]["label"] for pid in patient_ids]
    print(f"🔢 Total patients found: {len(patient_ids)}")

    # Create train-validation split (stratified by patient labels)
    train_pids, val_pids = train_test_split(
        patient_ids, test_size=VAL_SPLIT, stratify=patient_labels, random_state=42
    )

    # Assign samples to train and val based on patient IDs
    train_samples_fold = []
    val_samples_fold = []
    for pid in patient_ids:
        samples = patient_to_samples[pid]
        if pid in train_pids:
            train_samples_fold.extend(samples)
        else:
            val_samples_fold.extend(samples)

    print(f"🔢 Train samples: {len(train_samples_fold)}")
    print(f"🔢 Val samples: {len(val_samples_fold)}")

    # Compute mean and std from training samples
    temp_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    temp_dataset = ADNIDataset(samples=train_samples_fold, transform=temp_transform)
    mean, std = compute_mean_std(temp_dataset)

    def get_transforms(train=True):
        if train:
            return transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),
                transforms.Lambda(lambda x: x * torch.FloatTensor([torch.rand(1).item() * 0.2 + 0.9])),
                transforms.Normalize(mean=[mean], std=[std])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[mean], std=[std])
            ])

    # Create datasets
    train_dataset = ADNIDataset(samples=train_samples_fold, transform=get_transforms(train=True))
    val_dataset = ADNIDataset(samples=val_samples_fold, transform=get_transforms(train=False))
    test_dataset = ADNIDataset(samples=[
        {"path": os.path.join(DATA_DIR, "test", label_name, fname),
         "label": 1 if label_name == "AD" else 0,
         "patient_id": int(fname.replace(".jpeg", "").split("_")[0]),
         "scan_num": int(fname.replace(".jpeg", "").split("_")[1])}
        for label_name in ["AD", "NC"]
        for fname in os.listdir(os.path.join(DATA_DIR, "test", label_name))
        if fname.endswith(".jpeg")
    ], transform=get_transforms(train=False))

    print("Train label distribution:", Counter([s['label'] for s in train_dataset.samples]))
    print("Val label distribution:", Counter([s['label'] for s in val_dataset.samples]))
    print("Test label distribution:", Counter([s['label'] for s in test_dataset.samples]))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader