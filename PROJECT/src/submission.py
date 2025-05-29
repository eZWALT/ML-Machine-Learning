import pandas as pd
import numpy as np
from loguru import logger
from sklearn.base import ClassifierMixin
from loguru import logger

def evaluate_model(y_true, y_pred):
    logger.info("Print number of correct predictions:")
    correct_predictions = (y_true == y_pred).sum()
    final_score = correct_predictions / len(y_true)
    logger.success(f"Evaluation accuracy: {final_score:.4f}")
    return final_score


# if "language" column is not found then its the case of one-hot encoding (uni-model)
# if "language" is present then its an ensemble (multi-model)
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.base import ClassifierMixin

def evaluate_model(y_true, y_pred):
    logger.info("Print number of correct predictions:")
    correct_predictions = (y_true == y_pred).sum()
    final_score = correct_predictions / len(y_true)
    logger.success(f"Evaluation accuracy: {final_score:.4f}")
    return final_score


def generate_kaggle_submission(
    model,
    X_test: pd.DataFrame,
    test_meta: pd.DataFrame,
    output_path: str = "data/predictions_submission.csv",
    language_prefix: str = "language_",
    y_true: pd.Series = None,
    return_df: bool = False,
    verbose: bool = True
):

    if verbose:
        logger.info("Generating predictions...")

    # 1. Predict probabilities or binary outputs
    if hasattr(model, "predict_proba"):
        prob_predictions = model.predict_proba(X_test)
        root_probs = prob_predictions[:, 1]
    else:
        logger.warning("Model does not support predict_proba; using predict() directly.")
        root_probs = model.predict(X_test)

    # 2. Determine language column
    if 'language' in test_meta.columns:
        language = test_meta['language'].values
        logger.info("Detected multilingual (per-language model ensemble) setup.")
    else:
        language = X_test.filter(like=language_prefix).idxmax(axis=1).str.replace(language_prefix, '')
        logger.info("Detected unilingual (single model with one-hot languages) setup.")

    # 3. Compose initial prediction DataFrame
    preds = pd.DataFrame({
        'language': language,
        'sentence_id': test_meta['sentence_id'].values,
        'node': test_meta['node'].values,
        'root_prob': root_probs
    })

    # 4. Group and select best prediction per sentence
    idx_max_prob_per_group = preds.groupby(['language', 'sentence_id'], sort=False)['root_prob'].idxmax()
    preds_grouped = preds.loc[idx_max_prob_per_group, ['node']].copy()
    preds_grouped.rename(columns={'node': 'root'}, inplace=True)
    preds_grouped.reset_index(drop=True, inplace=True)
    preds_grouped['id'] = range(1, len(preds_grouped) + 1)
    preds_grouped = preds_grouped[['id', 'root']]

    # 5. Evaluate
    if y_true is not None:
        if verbose:
            logger.info("Evaluating predictions...")
        evaluate_model(y_true.reset_index(drop=True), preds_grouped['root'])

    # 6. Save or return
    if return_df:
        return preds_grouped

    preds_grouped.to_csv(output_path, index=False)
    if verbose:
        logger.success(f"Submission saved to: {output_path}")
