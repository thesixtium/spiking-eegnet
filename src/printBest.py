"""
Print the best trial's parameters from the Optuna SQLite study.

Usage:
    python3 print_best_params.py
    python3 print_best_params.py --db optuna_study.db --study snn_eegnet_v3_200_20
    python3 print_best_params.py --top 5          # also show top-N trials by value
"""
import argparse
import json
import optuna


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="../optuna_study.db",
                         help="Path to the SQLite DB file (default: optuna_study.db)")
    parser.add_argument("--study", default=None,
                         help="Study name. If omitted, auto-picks if there's only one "
                              "study in the DB, otherwise lists all available names.")
    parser.add_argument("--top", type=int, default=0,
                         help="Also print the top-N completed trials by value")
    args = parser.parse_args()

    storage = f"sqlite:///{args.db}"

    summaries = optuna.study.get_all_study_summaries(storage=storage)

    def _print_summaries():
        print(f"Found {len(summaries)} study(ies) in {args.db}:")
        for s in summaries:
            best = f"{s.best_trial.value:.4f}" if s.best_trial is not None else "n/a"
            print(f"  - {s.study_name:35s} trials={s.n_trials:<5d} best_value={best}")

    study_name = args.study
    if study_name is None:
        _print_summaries()
        if len(summaries) == 1:
            study_name = summaries[0].study_name
        else:
            print("\nMultiple (or zero) studies found in this DB — pass --study explicitly.")
            return
    elif study_name not in {s.study_name for s in summaries}:
        _print_summaries()
        print(f"\nStudy '{study_name}' not found in this DB — pick one of the names above.")
        return

    study = optuna.load_study(study_name=study_name, storage=storage)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"Study: {study_name}")
    print(f"  Total trials    : {len(study.trials)}")
    print(f"  Completed       : {len(completed)}")
    print(f"  Pruned          : {len(pruned)}")

    if not completed:
        print("\nNo completed trials yet — nothing to report.")
        return

    best = study.best_trial
    print(f"\nBest trial: #{best.number}")
    print(f"  Value (mean_bal_acc): {best.value:.4f}")
    print("  Params:")
    for k, v in sorted(best.params.items()):
        print(f"    {k:22s}: {v}")

    # Save to JSON alongside the DB for downstream use (e.g. --plot-best reruns)
    out_path = "best_params.json"
    with open(out_path, "w") as f:
        json.dump({"trial_number": best.number, "value": best.value,
                   "params": best.params}, f, indent=2)
    print(f"\nSaved to {out_path}")

    if args.top > 0:
        ranked = sorted(completed, key=lambda t: t.value, reverse=True)[: args.top]
        print(f"\nTop {len(ranked)} trials:")
        for t in ranked:
            print(f"  #{t.number:4d}  value={t.value:.4f}  params={t.params}")


if __name__ == "__main__":
    main()