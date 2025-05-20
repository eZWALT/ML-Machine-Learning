import numpy as np 
import pandas as pd 

# ===-----------------------------------------------------------------------===#
# Custom Loss Functions                                                        #
#                                                                              #
# In order to guide learning or evaluate the model, we use the official metric # 
# used to evaluate the model by the Kaggle competition
#                                                                              #
# Author: Walter Troiani                                                       #
# ===-----------------------------------------------------------------------===#


def correct_sentence_percentage(Y: pd.Series, Y_hat: pd.Series) -> float:
    """
    Compute the percentage of sentences for which all nodes with true label 1
    are correctly predicted as 1. Nodes with true label 0 are irrelevant.

    Assumes Y and Y_hat have the same index with sentence grouping (e.g., sentence_id in index level 0).

    Parameters:
        Y (pd.Series): True labels (0/1), indexed by sentence_id or MultiIndex.
        Y_hat (pd.Series): Predicted labels (0/1), same index as Y.

    Returns:
        float: Percentage of sentences meeting the criterion.
    """
    # Filter nodes where true label is 1 (these are important)
    important_nodes = (Y == 1)

    # For these nodes, check if predicted correctly (pred==1)
    correct_important = (Y_hat[important_nodes] == 1)

    # Group by sentence, check if all important nodes predicted correctly
    # Note: If a sentence has no nodes with true=1, consider it correct (or handle differently)
    per_sentence_correct = correct_important.groupby(level=0).all()

    # Sentences without any important nodes will be missing in per_sentence_correct,
    # we can consider those sentences as correct (or ignore them).
    # So fill missing sentences with True
    all_sentences = Y.index.get_level_values(0).unique()
    per_sentence_correct = per_sentence_correct.reindex(all_sentences, fill_value=True)

    # Calculate percentage
    pct = per_sentence_correct.mean()
    return pct
