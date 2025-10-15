import os
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Constants
DATA_DIR = "/home/groups/comp3710/ADNI/AD_NC"
IMG_SIZE = 224

# Preprocessing transforms
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

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
        return image, sample["label"]

# Data loader factory
def get_dataloaders(batch_size=32, num_workers=1):
    train_dataset = ADNIDataset(split="train", transform=get_transforms(train=True))
    test_dataset = ADNIDataset(split="test", transform=get_transforms(train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
