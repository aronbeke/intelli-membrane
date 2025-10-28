import sys
import os
import optuna
from optuna.trial import create_trial
from optuna.distributions import IntDistribution, CategoricalDistribution, FloatDistribution
import pandas as pd
import numpy as np
import torch
from pytorch_lightning import seed_everything
import random
import json

def set_random_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(seed, workers=True)

def load_preexisting_trials_from_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def run_bayesian_optimization(
        objective,
        n_trials=20,
        target_path = 'optuna_results.csv',
        check_existing = True, 
        search_space={},
        json_path="preexisting_trials.json"):
    """
    Runs Bayesian optimization and stores results for analysis.
    """

    if check_existing:
        preexisting_trials = load_preexisting_trials_from_json(json_path)

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=20)

    study = optuna.create_study(direction="minimize", pruner =pruner)

    if check_existing:
        for t in preexisting_trials:
            trial = create_trial(
                params=t["params"],
                distributions=search_space,
                value=t["value"]
            )
            study.add_trial(trial)

    study.optimize(objective, n_trials=n_trials, n_jobs = 1)

    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (test_loss): {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Save study results
    study.trials_dataframe().to_csv(target_path, index=False)

    return study

def log_in_json(trial, value, log_path):
    # Save results to JSON file
    result = {
        'trial_number': trial.number,
        'value': value,
        'params': trial.params
    }

    # Append to the JSON file
    if os.path.exists(log_path):
        with open(log_path, 'r+') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            data.append(result)
            f.seek(0)
            json.dump(data, f, indent=2)
    else:
        with open(log_path, 'w') as f:
            json.dump([result], f, indent=2)