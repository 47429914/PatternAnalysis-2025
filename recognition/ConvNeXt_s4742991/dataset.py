import os
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

# Constants
DATA_DIR = "/home/groups/comp3710/ADNI/AD_NC"
IMG_SIZE = 224

# Custom Dataset
class ADNIDataset(Dataset):
    def __init__(self, split="train", transform=None):
        assert split in ["train", "test"], "Split must be 'train' or 'test'"
        self.transform = transform
        self.samples = []

        split_dir = os.path.join(DATA_DIR, split)
        for label_name in ["AD", "NC"]:
            class_dir = os.path.join(split_dir, label_name)
            label = 1 if label_name == "AD" else 0
            for fname in os.listdir(class_dir):
                if fname.endswith(".jpeg"):
                    patient_id, scan_num = fname.replace(".jpeg", "").split("_")
                    path = os.path.join(class_dir, fname)
                    self.samples.append({
                        "path": path,
                        "label": label,
                        "patient_id": int(patient_id),
                        "scan_num": int(scan_num)
                    })

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
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=1)
    mean = 0.0
    std = 0.0
    total_images = 0
    print("🔄 Computing mean and std from training set...")
    for images, _, _ in loader:  # Ignore labels and patient IDs
        batch_samples = images.size(0)
        images = images.view(batch_samples, -1)
        mean += images.mean(dim=1).sum()
        std += images.std(dim=1).sum()
        total_images += batch_samples
    mean /= total_images
    std /= total_images
    print(f"✅ Computed mean: {mean.item()}")
    print(f"✅ Computed std: {std.item()}")
    return mean.item(), std.item()

# Data loader factory
def get_dataloaders(batch_size=32, num_workers=1):
    # Compute mean and std for single-channel images
    temp_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    temp_dataset = ADNIDataset(split="train", transform=temp_transform)
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

    train_dataset = ADNIDataset(split="train", transform=get_transforms(train=True))
    test_dataset = ADNIDataset(split="test", transform=get_transforms(train=False))

    print("Train label distribution:", Counter([s['label'] for s in train_dataset.samples]))
    print("Test label distribution:", Counter([s['label'] for s in test_dataset.samples]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
