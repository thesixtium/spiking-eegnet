import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score

@torch.no_grad()
def evaluate(model, loader, device, n_steps):
    """Returns balanced accuracy."""
    model.eval()
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        spk = model(xb, num_steps=n_steps)
        preds = spk.mean(0).argmax(1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(yb.numpy())
    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return balanced_accuracy_score(labels, preds)