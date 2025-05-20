from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd 
from typing import Callable
from loguru import logger
from src.unilingual_ensemble import UnilingualEnsembleClassifier
from sklearn.ensemble import RandomForestClassifier


def run_groupkfold_cv(X: pd.DataFrame, 
                      y: pd.Series, 
                      group_colname: str = "sentence_id",
                      clf_cls=UnilingualEnsembleClassifier,
                      clf_kwargs: dict = {
                        'base_model_cls': RandomForestClassifier,
                        'base_model_kwargs': {'n_estimators': 100, 'n_jobs': 1},
                        'language_colname': 'language',
                        'n_jobs': 4
                      },
                      n_splits: int = 5,
                      metric_fn: Callable = None,
                      verbose: bool = True):
    """
    Run GroupKFold cross-validation where all nodes from the same sentence (group) stay together.

    Parameters:
        X (pd.DataFrame): Feature matrix with group and language columns.
        y (pd.Series): Target labels.
        group_colname (str): Name of the column that groups rows by sentence.
        clf_cls (class): Classifier class to instantiate.
        clf_kwargs (dict): Keyword arguments to pass to the classifier.
        n_splits (int): Number of folds.
        random_state (int): For reproducibility.
        metric_fn (Callable): Evaluation function. Default is sklearn's accuracy_score.
        verbose (bool): Print fold-level results.

    Returns:
        List of scores (float) for each fold.
    """
    clf_kwargs = clf_kwargs or {}
    metric_fn = metric_fn or accuracy_score

    gkf = GroupKFold(n_splits=n_splits)
    groups = X[group_colname]
    scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        clf = clf_cls(**clf_kwargs)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)

        score = metric_fn(y_val, preds)
        scores.append(score)

        if verbose:
            logger.info(f"[Fold {fold + 1}] {metric_fn.__name__}: {score:.4f}")

    if verbose:
        logger.info(f"\nMean {metric_fn.__name__} over {n_splits} folds: {np.mean(scores):.4f}")

    return scores

if __name__ == "__main__":
    # === Synthetic data generation === #
    np.random.seed(42)
    n_sentences = 10000
    nodes_per_sentence = np.random.randint(3, 10, size=n_sentences)
    total_nodes = sum(nodes_per_sentence)

    sentence_ids = np.repeat(np.arange(n_sentences), nodes_per_sentence)
    languages = np.random.choice(['en', 'fr', 'de', 'es'], size=n_sentences)
    language_per_node = np.repeat(languages, nodes_per_sentence)

    # Features + target (binary: root node = 1, rest = 0)
    X = pd.DataFrame({
        'feature1': np.random.randn(total_nodes),
        'feature2': np.random.randn(total_nodes),
        'language': language_per_node,
        'sentence_id': sentence_ids
    })

    # Make root node the first one in each sentence group
    y = pd.Series(0, index=np.arange(total_nodes))
    sentence_starts = np.cumsum(np.concatenate(([0], nodes_per_sentence[:-1])))
    y.iloc[sentence_starts] = 1  # root = 1, others = 0

    # === Run CV === #
    accuracies = run_groupkfold_cv(
        X=X,
        y=y,
        group_colname='sentence_id',
        clf_cls=UnilingualEnsembleClassifier,
        clf_kwargs={
            'base_model_cls': RandomForestClassifier,
            'base_model_kwargs': {'n_estimators': 100, 'n_jobs': 1},
            'language_colname': 'language',
            'n_jobs': 4
        },
        n_splits=5,
        verbose=True
    )

    logger.success(f"\nFinal Mean Accuracy: {np.mean(accuracies):.4f}")
