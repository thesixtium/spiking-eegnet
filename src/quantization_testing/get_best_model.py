import optuna

storage = "sqlite:///optuna_study.db"

summaries = optuna.get_all_study_summaries(storage=storage)

for summary in summaries:
    print(f"\n{'='*50}")
    print(f"Study: {summary.study_name}")
    print(f"  Trials: {summary.n_trials}")

    study = optuna.load_study(study_name=summary.study_name, storage=storage)
    try:
        best = study.best_trial
        print(f"  Best trial: #{best.number}  acc={best.value:.4f}")
        for k, v in best.params.items():
            print(f"    {k}: {v}")
    except ValueError:
        print("  No completed trials yet.")