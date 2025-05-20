from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
import pandas as pd 
import numpy as np

# TODO Parallelize both training and inference 

class UnilingualEnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model_cls=RandomForestClassifier, base_model_kwargs=None, language_colname: str = "language", n_jobs: int = 1):
        self.base_model_cls = base_model_cls
        self.base_model_kwargs = base_model_kwargs or {}
        self.models_ = {}
        self.language_colname = language_colname
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Quality checks
        if isinstance(X, pd.DataFrame):
            if self.language_colname not in X.columns:
                raise ValueError(f"`{self.language_colname}` column must be present in X.")
            languages = X[self.language_colname].values
            X = X.drop(columns=[self.language_colname])
        else:
            raise ValueError("X must be a pandas DataFrame containing the language column.")
       
        # Fit all ensemble models for each language 
        self.models_ = {}
        for lang in np.unique(languages):
            idx = languages == lang 
            # Get the index of (X,Y) pairs 
            X_lang = X.loc[idx]
            Y_lang = y.loc[idx]
            
            # TODO: This is bad, this implies all models from the ensemble
            # must have the same hyperparameters which is not ideal but 
            # Greatly reduce the exploration space :)
            model = self.base_model_cls(**self.base_model_kwargs)
            model.fit(X_lang, Y_lang)
            self.models_[lang] = model 
        return self 
            
    # Given a Indicator matrix returns the (labels) of each value 
    def predict(self, X: pd.DataFrame) -> pd.Series:
        if isinstance(X, pd.DataFrame):
            if self.language_colname not in X.columns:
                raise ValueError(f"`{self.language_colname}` column must be present in X.")
            languages = X[self.language_colname].values
            X = X.drop(columns=[self.language_colname])
        else:
            raise ValueError("X must be a pandas DataFrame containing the language column.")

        preds = np.zeros(len(X), dtype=int)
        for lang in np.unique(languages):
            idx = languages == lang
            model = self.models_.get(lang)
            if model is None:
                raise ValueError(f"No model trained for language: {lang}")
            preds[idx] = model.predict(X.loc[idx])

        return preds
    # Given a Indicator matrix returns the (probs) of each value 
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        if isinstance(X, pd.DataFrame):
            if self.language_colname not in X.columns:
                raise ValueError(f"`{self.language_colname}` column must be present in X.")
            languages = X[self.language_colname].values
            X = X.drop(columns=[self.language_colname])
        else:
            raise ValueError("X must be a pandas DataFrame containing the language column.")

        probas = np.zeros((len(X), 2))  # Assuming binary classification
        for lang in np.unique(languages):
            idx = languages == lang
            model = self.models_.get(lang)
            if model is None:
                raise ValueError(f"No model trained for language: {lang}")
            probas[idx] = model.predict_proba(X.loc[idx])

        return probas            
    
    

from sklearn.metrics import accuracy_score
import time

if __name__ == "__main__":
    # Import your classifier from the module (or paste the class above here)
    # from your_module import UnilingualEnsembleClassifier

    # Create synthetic data
    np.random.seed(42)
    n_samples = 10000
    languages = np.random.choice(['en', 'fr', 'de', 'es'], size=n_samples)
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'language': languages
    })
    # Binary target depending loosely on feature1 + language
    y = (X['feature1'] + (X['language'] == 'en') * 0.5 + np.random.randn(n_samples) * 0.5 > 0).astype(int)

    # Split train/test (just a simple split here)
    train_frac = 0.8
    train_idx = np.random.choice(n_samples, int(n_samples*train_frac), replace=False)
    test_idx = np.setdiff1d(np.arange(n_samples), train_idx)

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # Instantiate the classifier with 4 parallel jobs
    clf = UnilingualEnsembleClassifier(n_jobs=4)

    # Measure training time
    start_train = time.time()
    clf.fit(X_train, y_train)
    end_train = time.time()

    print(f"Training time: {end_train - start_train:.2f} seconds")

    # Measure inference time
    start_pred = time.time()
    preds = clf.predict(X_test)
    end_pred = time.time()

    print(f"Inference time: {end_pred - start_pred:.2f} seconds")

    # Accuracy check
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc:.4f}")
