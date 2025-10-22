import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
N_FOLDS = 5

def train():
    print(f"Device: {DEVICE}")
    
    # Store results for each fold
    fold_val_accuracies = []
    
    # Cross-validation loop
    for fold in range(N_FOLDS):
        print(f"\n===== Fold {fold + 1}/{N_FOLDS} =====")
        # Load data for this fold (train on 4 folds, validate on 1)
        train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE, fold=fold)
        
        # Model (reset for each fold)
        model = get_model().to(DEVICE)
        print(f"Trainable parameters: {count_parameters(model):,}")

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        best_acc = 0.0
        current_patience = 0
        best_model_path = f"best_model_fold{fold}.pth"

        for epoch in range(EPOCHS):
            # Training phase
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

            # Validation phase
            model.eval()
            patient_probs = defaultdict(list)
            patient_labels = {}

            with torch.no_grad():
                for images, labels, patient_ids in val_loader:
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
            val_acc = correct / len(final_preds) if len(final_preds) > 0 else 0.0
            scheduler.step(val_acc)
            print(f"Fold {fold + 1} | Epoch {epoch + 1:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            # Save best model for this fold
            if val_acc > best_acc:
                best_acc = val_acc
                current_patience = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"✅ Best model for fold {fold + 1} updated (Val Acc: {best_acc:.4f}).")
            else:
                current_patience += 1
                if current_patience > MAX_PATIENCE:
                    print(f"Validation accuracy not improving for fold {fold + 1}: Ending training")
                    break

        # Store best validation accuracy for this fold
        fold_val_accuracies.append(best_acc)
        print(f"Best validation accuracy for fold {fold + 1}: {best_acc:.4f}")

    # Compute and print average accuracy across folds
    avg_val_acc = np.mean(fold_val_accuracies)
    print(f"\n===== Final Results =====")
    print(f"Accuracy for each fold: {[f'{acc:.4f}' for acc in fold_val_accuracies]}")
    print(f"Average Validation Accuracy across {N_FOLDS} folds: {avg_val_acc:.4f}")

if __name__ == "__main__":
    train()