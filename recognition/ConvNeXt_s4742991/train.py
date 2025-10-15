import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import get_dataloaders
from modules import get_model, count_parameters
import numpy as np
from collections import defaultdict


# Training config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_PATIENCE = 10

def train():
    # Load data
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    current_patience = 0

    print(f"Device: {DEVICE}")

    # Model
    model = get_model().to(DEVICE)
    print(f"Trainable parameters: {count_parameters(model):,}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in train_loader:
            images = images.to(DEVICE)
            labels = labels.float().to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = total_loss / total
        scheduler.step(val_acc)
    
        # Validation
        model.eval()
        patient_probs = defaultdict(list)
        patient_labels = {}

        with torch.no_grad():
            for images, labels, patient_ids in test_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE).long()
                outputs = model(images)
                probs = torch.sigmoid(outputs)

                for pid, prob, label in zip(patient_ids, probs, labels):
                    pid = pid.item()
                    patient_probs[pid].append(prob.item())
                    patient_labels[pid] = label.item()

        # Aggregate predictions per patient
        final_preds = {pid: int(np.mean(probs) > 0.5) for pid, probs in patient_probs.items()}
        final_labels = patient_labels

        # Compute patient-level accuracy
        correct = sum(final_preds[pid] == final_labels[pid] for pid in final_preds)
        val_acc = correct / len(final_preds)
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            current_patience = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Best model updated.")
        else:
            if (current_patience > MAX_PATIENCE):
                print("Validation accuracy not improving: Ending training to save resources")
                break
            else:
                current_patience += 1

if __name__ == "__main__":
    train()
