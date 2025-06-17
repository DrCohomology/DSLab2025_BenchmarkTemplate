"""
Run a dummy experiment.
"""

import numpy as np
import pandas as pd

from itertools import product
from joblib import Parallel, delayed
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.model_selection import StratifiedKFold

from sklearn.tree import DecisionTreeClassifier


# --- config.py
# Macros are to be defined in config.py (they are here now for convenience)
BASE_DIR = Path(".")
RESULT_DIR = BASE_DIR / "dummy_results"
RESULT_DIR.mkdir(exist_ok=True)

# Define other parameters
SEED = 1444
CV = StratifiedKFold(n_splits=5, random_state=SEED - 1, shuffle=True)

# Define what we are benchmarking
MODELS = [DecisionTreeClassifier(random_state=SEED),
          RandomForestClassifier(random_state=SEED + 1),]

# Define the experimental conditions
DATASETS = [1, 2, 3]  # in the real experiments, these are valid ids of OpenML datasets
SCORINGS = [matthews_corrcoef, accuracy_score]
CONDITIONS = product(DATASETS, SCORINGS)

# --- main.py
# This part is the true main
rng = np.random.default_rng(SEED)

# "Run" the CONDITIONS: "evaluate" all models under the same condition
def main_loop(result_dir, models, dataset, scoring, cv, seed):
    """
    Refer to `main.py` for a proper example of this function
    """

    flag_fail = 0

    exp_name = f"{dataset}_{scoring.__name__}_{seed}"

    out = []
    for cv_fold in range(5):
        for model in models:
            try:
                evaluation = np.random.random()
            except:
                evaluation = np.nan
                flag_fail = 1

            out.append({
                "dataset": dataset,
                "scoring": scoring.__name__,
                "seed": seed,
                "model": model.__class__.__name__,
                "cv_fold": cv_fold,
                "evaluation": evaluation
            })
    out = pd.DataFrame(out)
    # out.to_parquet(RESULT_DIR / f"{exp_name}.parquet")
    out.to_csv(RESULT_DIR / f"{exp_name}.csv", index=False)

    return flag_fail


if __name__ == "__main__":
    Parallel(n_jobs=1, verbose=0)(
        delayed(main_loop)(RESULT_DIR, MODELS, dataset, scoring, CV, SEED)
        for (idx, (dataset, scoring)) in enumerate(CONDITIONS)
    )




