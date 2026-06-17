import optuna

storage = "sqlite:///optuna_study.db"
divider = "=" * 50

summaries = optuna.get_all_study_summaries(storage=storage)

for summary in summaries:
    print("\n" + divider)
    print("Study: " + summary.study_name)
    print("  Trials: " + str(summary.n_trials))

    study = optuna.load_study(study_name=summary.study_name, storage=storage)
    try:
        best = study.best_trial
        print("  Best trial: #" + str(best.number) + "  acc=" + str(round(best.value, 4)))
        for k, v in best.params.items():
            print("    " + k + ": " + str(v))
    except ValueError:
        print("  No completed trials yet.")