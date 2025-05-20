from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


# TODO: Revise this, this shit was done by the lil generative fella

def evaluate_model_with_groupkfold(
    df: pd.DataFrame,
    label_col: str,
    group_col: str,
    categorical_cols: list,
    model=None,
    n_splits: int = 5
):
    """
    Perform grouped K-fold cross-validation.
    
    Parameters:
        df (pd.DataFrame): The full training dataframe.
        label_col (str): Name of the target/label column (e.g., "root").
        group_col (str): Column used for grouping (e.g., "sentence_id").
        categorical_cols (list): List of categorical columns to one-hot encode (e.g., ["language"]).
        model: Any scikit-learn model (default: RandomForest).
        n_splits (int): Number of cross-validation folds.

    Returns:
        list: Accuracy scores per fold
    """
    if model is None:
        model = RandomForestClassifier()

    gkf = GroupKFold(n_splits=n_splits)
    groups = df[group_col]
    y = df[label_col]

    fold_accuracies = []
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, y, groups)):
        print(f"\n🔁 Fold {fold + 1}")

        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        # One-hot encoding
        enc.fit(train_df[categorical_cols])
        X_train_cat = pd.DataFrame(enc.transform(train_df[categorical_cols]),
                                   columns=enc.get_feature_names_out(),
                                   index=train_df.index)
        X_val_cat = pd.DataFrame(enc.transform(val_df[categorical_cols]),
                                 columns=enc.get_feature_names_out(),
                                 index=val_df.index)

        # Drop old categorical columns and insert encoded ones
        train_df = pd.concat([train_df.drop(columns=categorical_cols + [label_col]), X_train_cat], axis=1)
        val_df = pd.concat([val_df.drop(columns=categorical_cols + [label_col]), X_val_cat], axis=1)

        X_train, y_train = train_df, df[label_col].iloc[train_idx]
        X_val, y_val = val_df, df[label_col].iloc[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        acc = accuracy_score(y_val, preds)
        print(f"✅ Fold {fold + 1} Accuracy: {acc:.4f}")
        print(classification_report(y_val, preds, digits=4))
        fold_accuracies.append(acc)

    print(f"\n🎯 Average Accuracy across {n_splits} folds: {np.mean(fold_accuracies):.4f}")
    return fold_accuracies



if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import GroupKFold
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([1, 2, 3, 4, 5, 6])
    groups = np.array([0, 0, 2, 2, 3, 3])
    group_kfold = GroupKFold(n_splits=2)
    group_kfold.get_n_splits(X, y, groups)
    print(group_kfold)
    for i, (train_index, test_index) in enumerate(group_kfold.split(X, y, groups)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}, group={groups[train_index]}")
        print(f"  Test:  index={test_index}, group={groups[test_index]}")