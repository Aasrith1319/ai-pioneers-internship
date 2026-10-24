"""
Week 4 - End-to-end training pipeline for deployment.

Wraps every preprocessing step from Week 1 and the tuned model from Week 3 into a
single scikit-learn Pipeline, so the deployed API accepts raw, human-readable input
(e.g. sex="female", embarked="C") and applies exactly the same transformations
used during training. The fitted pipeline is serialized with joblib.
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEED = 42

from features import FeatureEngineer, RAW_FEATURES


def build_pipeline():
    numeric = ["age", "fare", "fare_per_person", "parch", "family_size"]
    passthrough = ["pclass", "sex", "has_deck"]
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))]), ["embarked"]),
        ("pass", "passthrough", passthrough),
    ])
    model = RandomForestClassifier(n_estimators=300, max_depth=6, max_features=0.5,
                                   min_samples_leaf=1, random_state=SEED)   # tuned in Week 3
    return Pipeline([("features", FeatureEngineer()), ("preprocess", pre), ("model", model)])


def main():
    raw = pd.read_csv(ROOT / "data" / "titanic_raw.csv")
    raw["has_deck"] = raw["deck"].notna().astype(int)
    X, y = raw[RAW_FEATURES], raw["survived"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    pipe = build_pipeline().fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {"accuracy": round(accuracy_score(y_te, pred), 3),
               "f1": round(f1_score(y_te, pred), 3),
               "roc_auc": round(roc_auc_score(y_te, proba), 3)}
    print("Hold-out metrics:", metrics)

    # Refit on all data for the production artifact
    pipe = build_pipeline().fit(X, y)
    (HERE / "model").mkdir(exist_ok=True)
    joblib.dump(pipe, HERE / "model" / "titanic_pipeline.joblib")
    meta = {"model": "RandomForestClassifier (tuned)", "features": RAW_FEATURES,
            "holdout_metrics": metrics, "sklearn_version": sklearn.__version__,
            "trained_rows": len(X)}
    (HERE / "model" / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("Saved model/titanic_pipeline.joblib")


if __name__ == "__main__":
    main()
