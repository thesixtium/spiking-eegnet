"""
Delete a study BY NAME from an Optuna SQLite database.

Edit DB_PATH and STUDY_NAME below, then run:
    python3 delete_study.py
"""
import optuna

DB_PATH = "../optuna_study.db"
STUDY_NAME = "snn_eegnet_v3_200_20"


def main():
    storage = f"sqlite:///{DB_PATH}"

    summaries = optuna.study.get_all_study_summaries(storage=storage)
    names = {s.study_name for s in summaries}

    def _print_summaries():
        print(f"Found {len(summaries)} study(ies) in {DB_PATH}:")
        for s in summaries:
            best = f"{s.best_trial.value:.4f}" if s.best_trial is not None else "n/a"
            print(f"  - {s.study_name:35s} trials={s.n_trials:<5d} best_value={best}")

    if STUDY_NAME not in names:
        _print_summaries()
        print(f"\nStudy '{STUDY_NAME}' not found in this DB — pick one of the names above.")
        return

    # Show what we're about to delete
    target = next(s for s in summaries if s.study_name == STUDY_NAME)
    best = f"{target.best_trial.value:.4f}" if target.best_trial is not None else "n/a"
    print(f"About to delete study: {STUDY_NAME}")
    print(f"  Trials     : {target.n_trials}")
    print(f"  Best value : {best}")
    print(f"  DB         : {DB_PATH}")

    confirm = input(f"\nType the study name to confirm deletion: ").strip()
    if confirm != STUDY_NAME:
        print("Confirmation did not match — aborting, nothing deleted.")
        return

    optuna.delete_study(study_name=STUDY_NAME, storage=storage)
    print(f"\nDeleted study '{STUDY_NAME}' from {DB_PATH}.")

    remaining = optuna.study.get_all_study_summaries(storage=storage)
    print(f"Studies remaining: {len(remaining)}")
    for s in remaining:
        print(f"  - {s.study_name}")


if __name__ == "__main__":
    main()