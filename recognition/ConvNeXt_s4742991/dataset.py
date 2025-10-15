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
        image = Image.open(sample["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, sample["label"], sample["patient_id"]

# Compute mean and std from training set
def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=1)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("🔄 Computing mean and std from training set...")
    for images, _ in loader:
        for i in range(3):
            mean[i] += images[:, i, :, :].mean()
            std[i] += images[:, i, :, :].std()
    mean /= len(loader)
    std /= len(loader)
    print(f"✅ Computed mean: {mean.tolist()}")
    print(f"✅ Computed std: {std.tolist()}")
    return mean.tolist(), std.tolist()

# Data loader factory
def get_dataloaders(batch_size=32, num_workers=1):
    # Temporary transform for computing stats
    temp_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    # temp_dataset = ADNIDataset(split="train", transform=temp_transform)
    # mean, std = compute_mean_std(temp_dataset)
    # Final transforms with computed stats
    def get_transforms(train=True):
        if train:
            return transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1156277284026146, 0.1156277284026146, 0.1156277284026146], 
                                     std=[0.22283731400966644, 0.22283731400966644, 0.22283731400966644])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1156277284026146, 0.1156277284026146, 0.1156277284026146], 
                                     std=[0.22283731400966644, 0.22283731400966644, 0.22283731400966644])
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
