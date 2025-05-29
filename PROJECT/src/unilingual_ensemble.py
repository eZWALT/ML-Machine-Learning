from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# ===-----------------------------------------------------------------------===#
# Unilingual Ensemble Classifier                                               #
#                                                                              #
# Generic abstract ensemble that classifies a target variable Y but using a    #
# different base classifier for each language. Can be parallelized by using the#
# n_jobs hyperparameter. The hypothesis is that this works better than a multi-#
#-lingual model (many languages one model) due to the simplicity of classic ml #
# we also incorporated GridSearchCV                                            #
#                                                                              #
# Author: Walter Troiani                                                       #
# ===-----------------------------------------------------------------------===#


class UnilingualEnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        base_model_cls=RandomForestClassifier,
        base_model_kwargs=None,
        language_colname: str = "language",
        n_jobs: int = 1,
        cv: int = 2,
        gridsearch_per_language: bool = False,
        param_grid: dict = None,
    ):
        self.base_model_cls = base_model_cls
        self.base_model_kwargs = base_model_kwargs or {}
        self.language_colname = language_colname
        self.n_jobs = n_jobs
        self.models_ = {}
        self.cv = cv
        self.gridsearch_per_language = gridsearch_per_language
        self.param_grid = param_grid

    def _fit_one(self, lang, X_lang, y_lang):
        base_model = self.base_model_cls(**self.base_model_kwargs)
        if self.gridsearch_per_language:
            if self.param_grid is None:
                raise ValueError("`param_grid` must be provided if gridsearch_per_language=True")
            search = GridSearchCV(base_model, self.param_grid, cv=self.cv, n_jobs=1)
            search.fit(X_lang, y_lang)
            model = search.best_estimator_
        else:
            base_model.fit(X_lang, y_lang)
            model = base_model
        return lang, model

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.language_colname not in X.columns:
            raise ValueError(f"`{self.language_colname}` column must be present in X.")
        languages = X[self.language_colname].values
        X_ = X.drop(columns=[self.language_colname])
        
        unique_languages = np.unique(languages)
        n_langs = len(unique_languages)
        if self.n_jobs > n_langs:
            raise ValueError(f"n_jobs ({self.n_jobs}) must be ≤ number of unique languages ({n_langs}).")

        grouped_data = {
            lang: (X_.loc[languages == lang], y.loc[languages == lang])
            for lang in np.unique(languages)
        }

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_one)(lang, X_lang, y_lang)
            for lang, (X_lang, y_lang) in grouped_data.items()
        )

        self.models_ = dict(results)
        return self

    def _predict_one(self, lang, model, X_lang):
        return lang, model.predict(X_lang)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.language_colname not in X.columns:
            raise ValueError(f"`{self.language_colname}` column must be present in X.")
        languages = X[self.language_colname].values
        X_ = X.drop(columns=[self.language_colname])

        grouped_data = {
            lang: X_.loc[languages == lang]
            for lang in np.unique(languages)
        }

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._predict_one)(lang, self.models_[lang], X_lang)
            for lang, X_lang in grouped_data.items()
        )

        preds = np.zeros(len(X), dtype=int)
        for lang, pred in results:
            idx = languages == lang
            preds[idx] = pred

        return preds

    def _predict_proba_one(self, lang, model, X_lang):
        return lang, model.predict_proba(X_lang)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.language_colname not in X.columns:
            raise ValueError(f"`{self.language_colname}` column must be present in X.")
        languages = X[self.language_colname].values
        X_ = X.drop(columns=[self.language_colname])

        grouped_data = {
            lang: X_.loc[languages == lang]
            for lang in np.unique(languages)
        }

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._predict_proba_one)(lang, self.models_[lang], X_lang)
            for lang, X_lang in grouped_data.items()
        )

        # Dynamically detect number of classes from one of the models
        n_classes = list(self.models_.values())[0].n_classes_
        probas = np.zeros((len(X), n_classes))

        for lang, proba in results:
            idx = languages == lang
            probas[idx] = proba

        return probas

        
    
    

from sklearn.metrics import accuracy_score
import time

if __name__ == "__main__":
    # Generate synthetic dataset
    np.random.seed(42)
    n_samples = 100000
    languages = np.random.choice(['en', 'fr', 'de', 'es'], size=n_samples)
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'language': languages
    })
    y = (X['feature1'] + 
         (X['language'] == 'en') * 0.5
         + np.random.randn(n_samples) * 0.5 > 0).astype(int)

    # Train/test split
    train_frac = 0.8
    train_idx = np.random.choice(n_samples, int(n_samples * train_frac), replace=False)
    test_idx = np.setdiff1d(np.arange(n_samples), train_idx)
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]


    def benchmark(n_jobs):
        clf = UnilingualEnsembleClassifier(n_jobs=n_jobs, base_model_cls=RandomForestClassifier, base_model_kwargs={"n_jobs": 1})
        times = {}

        start = time.time()
        clf.fit(X_train, y_train)
        times['fit_time'] = time.time() - start

        start = time.time()
        preds = clf.predict(X_test)
        times['predict_time'] = time.time() - start
        times['accuracy'] = accuracy_score(y_test, preds)

        start = time.time()
        _ = clf.predict_proba(X_test)
        times['predict_proba_time'] = time.time() - start

        return times


    results = {
        'n_jobs=1': benchmark(n_jobs=1),
        'n_jobs=4': benchmark(n_jobs=4)
    }
    print(results)