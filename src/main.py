"""
This is the main file for your pipeline.
It should load the configuration of the experiments from config.py (macros, objects) and then run the pipeline

Refer to `create_dummy_results.py` to see the format of results.
"""

# Imports
from joblib import Parallel, delayed
from openml.datasets import get_dataset
from openml.exceptions import OpenMLServerException
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from stopit import ThreadingTimeout as Timeout

import src.config as cfg


def main_loop(result_dir, models, ...):
    """
    Run an experiment: run all models under one experimental condition.
    Store the results in an appropriate directory for later use.

    Inputs
        - result_dir: directory where results are stored
        - condition: an experimental condition (tuple of objects loaded from config.py)
            - in your function, list all the factors as inputs

    Example
        main_loop(result_dir=RESULT_DIR, dataset=143, model=LGBMClassifier(), encoder=OneHotEncoder(),
                  scoring=roc_auc_score)
    """

    # Define the pipeline
    cats = X.select_dtypes(include=("category", "object")).columns
    nums = X.select_dtypes(exclude=("category", "object")).columns
    catpipe = Pipeline([()])
    numpipe = Pipeline([()])
    prepipe = ColumnTransformer([("cat", catpipe, cats), ("num", numpipe, nums)], remainder="passthrough")

    # `dataset` is an openml object
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )

    # Run the experiment (with timeout)
    cv = StratifiedKFold(n_splits=n_splits, random_state=random_state)
    out = {} # it will contain the results of the experiments. Initialize it correctly.
    try:
        with Timeout(timeout, swallow_exc=False) as timeout_ctx:
            for fold, (tr, te) in enumerate(cv.split(X, y)):
                Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
                Xtr, ytr = Xtr.reset_index(drop=True), ytr.reset_index(drop=True)
                Xte, yte = Xte.reset_index(drop=True), yte.reset_index(drop=True)

                XEtr = prepipe.fit_transform(Xtr, ytr)
                XEte = prepipe.transform(Xte)

                for model in models:
                    # Train model on XEtr, ytr
                    # Predict XEte
                    # Evaluate prediction against yte
                    # Append result to out
    except Exception as error:
        # Handle exceptions properly without the file stopping
    else:
        # Everything ran smoothly, save the results in a parquet file in cfg.RESULT_DIR, with a unique identifier

    return # return does not have to be `None`, you can use as feedback


if __name__ == "__main__":

    # Create the necessary directories
    # Define the experiments. Each experiment is defined by an experimental condition.
    #   each condition is a tuple of objects
    CONDITIONS = itertools.product(cfg.DATASET_NAMES, ...)

    # We are benchmarking models, they are not an experimental condition and are separate
    MODELS = cfg.MODELS

    # Load datasets from openml
    datasets = {}
    for dname, did in zip(cfg.DATASET_NAMES, cfg.DATASET_IDS):
        try:
            datasets[dname] = get_dataset(did)
        except OpenMLServerException:
            # Handle the exception appropriately

    CONDITIONS = [
        (datasets[dname], ...)
        for (dname, ...) in CONDITIONS
    ]

    # Run
    Parallel(n_jobs=cfg.NUM_PROCESSES, verbose=0)(
        delayed(main_loop)(cfg.RESULT_DIR, MODELS, CONDITIONS, ...)
        for (_, ...) in enumerate(CONDITIONS)
    )
