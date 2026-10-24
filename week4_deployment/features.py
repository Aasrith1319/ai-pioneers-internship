"""Custom feature-engineering step shared by training and the API.

Kept in its own module so the joblib-serialized pipeline can be loaded by app.py.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

RAW_FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "has_deck"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds the engineered features from Week 1 and caps fare outliers (IQR)."""

    def fit(self, X, y=None):
        q1, q3 = X["fare"].quantile([0.25, 0.75])
        self.fare_cap_ = q3 + 1.5 * (q3 - q1)
        return self

    def transform(self, X):
        X = X.copy()
        X["sex"] = X["sex"].map({"male": 0, "female": 1})
        X["family_size"] = X["sibsp"] + X["parch"] + 1
        X["fare"] = X["fare"].clip(upper=self.fare_cap_)
        X["fare_per_person"] = X["fare"] / X["family_size"]
        return X.drop(columns=["sibsp"])
