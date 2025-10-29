import torch
import numpy as np
from collections import defaultdict
from dataset import get_dataloaders
from modules import get_model

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"
THRESHOLD = 0.5

def predict():
    # Load test data
    _, _, test_loader = get_dataloaders(batch_size=32)

    # Load model
    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Store slice-level predictions grouped by patient
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
    final_preds = {pid: int(np.mean(probs) > THRESHOLD) for pid, probs in patient_probs.items()}
    final_labels = patient_labels

    # Compute patient-level accuracy
    correct = sum(final_preds[pid] == final_labels[pid] for pid in final_preds)
    total = len(final_preds)
    acc = correct / total
    print(f"✅ Patient-Level Accuracy: {acc:.4f}")

    # Optional: show a few predictions
    for pid in list(final_preds.keys())[:10]:
        print(f"Patient {pid} → Predicted: {final_preds[pid]} | Actual: {final_labels[pid]}")

if __name__ == "__main__":
    predict()
