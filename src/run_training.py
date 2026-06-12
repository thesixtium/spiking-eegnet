import torch.nn as nn
import torch.optim as optim

from train_one_epoch import train_one_epoch
from evaluate import evaluate

def run_training(
    model, train_loader, val_loader,
    epochs, lr, device,
    n_steps_train=4, n_steps_eval=10,
    eval_every_epoch=True,
):
    """
    Train model; return history dict {loss, bal_acc}.

    Parameters
    ----------
    eval_every_epoch : bool
        If True, evaluate on val_loader after every epoch (train/test split mode).
        If False, only record training loss per epoch and evaluate once at the end
        (LOSO mode — avoids leaking test-subject info during training).
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "bal_acc": []}

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, n_steps_train)
        history["loss"].append(loss)

        if eval_every_epoch:
            bal_acc = evaluate(model, val_loader, device, n_steps_eval)
            history["bal_acc"].append(bal_acc)
            print(f"  epoch {epoch:3d}/{epochs}  loss={loss:.4f}  bal_acc={bal_acc:.4f}")
        else:
            print(f"  epoch {epoch:3d}/{epochs}  loss={loss:.4f}")

    if not eval_every_epoch:
        final_acc = evaluate(model, val_loader, device, n_steps_eval)
        history["bal_acc"] = [final_acc]   # single entry — evaluated once at end
        print(f"  Final balanced accuracy (held-out test): {final_acc:.4f}")

    return history