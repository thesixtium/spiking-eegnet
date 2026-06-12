from make_loader import make_loader
from build_model import build_model
from run_training import run_training

def experiment_loso(
    X, y, subject_ids, meta, device, cfg,
    test_subject_idx: int = 0,
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

    model = build_model(meta, device)
    history = run_training(
        model, train_loader, val_loader,
        epochs=cfg["epochs"], lr=cfg["lr"], device=device,
        n_steps_train=cfg["n_steps_train"], n_steps_eval=cfg["n_steps_eval"],
        eval_every_epoch=True,
    )
    final_acc = history["bal_acc"][-1]
    return history, final_acc