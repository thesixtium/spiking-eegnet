import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from train_one_epoch import aggregate_logits


@torch.no_grad()
def evaluate(model, loader, device, n_steps, readout_mode="spk_mean"):
    """Returns balanced accuracy."""
    model.eval()
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        spk, mem = model(xb, num_steps=n_steps)
        if readout_mode == "mem_last":
            logits = model.classifier(mem[-1].flatten(1))
        else:
            logits = aggregate_logits(spk, mem, readout_mode)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(yb.numpy())
    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return balanced_accuracy_score(labels, preds)