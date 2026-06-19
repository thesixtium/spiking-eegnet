# experiment_loso.py
import optuna

from make_loader import make_loader
from build_model import build_model
from run_training import run_training


def experiment_loso(
    X, y, subject_ids, meta, device, cfg,
    test_subject_idx: int = 0,
    model_kwargs: dict = None,
):
    """
    Leave-One-Subject-Out: hold out subject `test_subject_idx`, train on rest.
    Evaluates on the held-out subject every epoch.
    """
    print(f"\n=== LOSO (hold out subject {test_subject_idx}) ===")
    test_mask  = subject_ids == test_subject_idx
    train_mask = ~test_mask

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask],  y[test_mask]
    print(f"  Train: {X_tr.shape[0]} trials  |  Test: {X_te.shape[0]} trials")

    train_loader = make_loader(X_tr, y_tr, cfg["batch_size"])
    val_loader   = make_loader(X_te, y_te, cfg["batch_size"], shuffle=False)

    model = build_model(meta, device, **(model_kwargs or {}))
    history = run_training(
        model, train_loader, val_loader,
        epochs=cfg["epochs"], lr=cfg["lr"], device=device,
        n_steps_train=cfg["n_steps_train"], n_steps_eval=cfg["n_steps_eval"],
        eval_every_epoch=True,
        patience=cfg.get("patience"),
        trial=cfg.get("trial"),
    )
    final_acc = history["bal_acc"][-1]
    return history, final_acc, model


def experiment_loso_all(
    X, y, subject_ids, meta, device, cfg,
    model_kwargs: dict = None,
    trial=None,
):
    """
    True Leave-One-Subject-Out: run a full per-subject LOSO experiment
    for every subject in the dataset (9x compute for BNCI2014_001).

    Pruning is evaluated once PER SUBJECT (not per epoch): after each
    subject finishes training, we report the running mean balanced
    accuracy across all subjects evaluated so far. If that running mean
    is too low, Optuna prunes the trial and we skip the remaining
    subjects entirely -- the idea being that one weak subject is enough
    to drag down the whole model's average, so there's no point paying
    for the rest.

    `cfg` should NOT contain a 'trial' key (or it should be None) --
    per-epoch pruning inside `run_training` is disabled here in favor
    of this per-subject pruning loop, since reporting at two different
    granularities to the same trial would collide.
    """
    subjects = sorted(set(int(s) for s in subject_ids))
    n_subjects = len(subjects)

    histories = {}
    accs = []

    for i, subj in enumerate(subjects):
        history, final_acc, _model = experiment_loso(
            X, y, subject_ids, meta, device, cfg,
            test_subject_idx=subj,
            model_kwargs=model_kwargs,
        )
        histories[subj] = history
        accs.append(final_acc)

        running_mean = sum(accs) / len(accs)
        print(f"  [LOSO {i + 1}/{n_subjects}] subject {subj}: "
              f"acc={final_acc:.4f}  running_mean={running_mean:.4f}")

        if trial is not None:
            trial.report(running_mean, step=i)
            if trial.should_prune():
                print(f"  [pruned by Optuna after subject {subj} "
                      f"({i + 1}/{n_subjects} subjects), "
                      f"running_mean={running_mean:.4f}]")
                raise optuna.exceptions.TrialPruned()

    mean_acc = sum(accs) / len(accs)
    return histories, accs, mean_acc