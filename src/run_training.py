# run_training.py
import optuna
import torch.nn as nn
import torch.optim as optim

from train_one_epoch import train_one_epoch
from evaluate import evaluate


def run_training(
    model, train_loader, val_loader,
    epochs, lr, device,
    n_steps_train=4, n_steps_eval=10,
    readout_mode="spk_mean",
    eval_every_epoch=True,
    patience=None,
    trial=None,
):
    """
    Train model; return history dict {loss, bal_acc}.

    Parameters
    ----------
    readout_mode : str
        How to collapse timestep outputs into per-sample logits.
        One of: 'spk_mean', 'spk_last', 'spk_sum', 'mem_last'.
        Passed through to train_one_epoch() and evaluate().
    eval_every_epoch : bool
        If True, evaluate on val_loader after every epoch and monitor
        balanced accuracy for early stopping.
        If False, only record training loss per epoch and evaluate once
        at the end (LOSO mode). Early stopping monitors loss in this case.
    patience : int or None
        Stop training if the monitored metric does not improve for this
        many consecutive epochs. None disables early stopping.
        When eval_every_epoch=True,  monitors balanced accuracy (higher = better).
        When eval_every_epoch=False, monitors training loss  (lower  = better).
    trial : optuna.Trial or None
        If provided, reports intermediate bal_acc each epoch and raises
        TrialPruned if Optuna decides to prune.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "bal_acc": []}

    best_metric   = None
    epochs_no_imp = 0

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device,
                               n_steps_train, readout_mode=readout_mode)
        history["loss"].append(loss)

        if eval_every_epoch:
            bal_acc = evaluate(model, val_loader, device, n_steps_eval,
                               readout_mode=readout_mode)
            history["bal_acc"].append(bal_acc)
            print(f"  epoch {epoch:3d}/{epochs}  loss={loss:.4f}  bal_acc={bal_acc:.4f}", end="")

            # ── Optuna pruning ────────────────────────────────────────
            if trial is not None:
                trial.report(bal_acc, epoch)
                if trial.should_prune():
                    print(f"  [pruned by Optuna at epoch {epoch}]")
                    raise optuna.exceptions.TrialPruned()

            if patience is not None:
                improved = best_metric is None or bal_acc > best_metric
                if improved:
                    best_metric   = bal_acc
                    epochs_no_imp = 0
                else:
                    epochs_no_imp += 1
                if epochs_no_imp >= patience:
                    print(f"  [early stop: no bal_acc improvement for {patience} epochs]")
                    break

            print()

        else:
            print(f"  epoch {epoch:3d}/{epochs}  loss={loss:.4f}", end="")

            if patience is not None:
                improved = best_metric is None or loss < best_metric
                if improved:
                    best_metric   = loss
                    epochs_no_imp = 0
                else:
                    epochs_no_imp += 1
                if epochs_no_imp >= patience:
                    print(f"  [early stop: no loss improvement for {patience} epochs]")
                    break

            print()

    if not eval_every_epoch:
        final_acc = evaluate(model, val_loader, device, n_steps_eval,
                             readout_mode=readout_mode)
        history["bal_acc"] = [final_acc]
        print(f"  Final balanced accuracy (held-out test): {final_acc:.4f}")

    return history