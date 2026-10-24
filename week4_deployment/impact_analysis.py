"""
Week 4 - Impact analysis for the deployed pipeline.

  1. Ablation: adds pipeline steps one at a time and measures 5-fold CV accuracy / ROC-AUC
  2. Concrete example predictions for five representative passengers
  3. Deployment performance: model size, load time, single and batch API latency
  4. Validation examples: what the API returns for bad input
"""
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from features import FeatureEngineer, RAW_FEATURES
from train import build_pipeline

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEED = 42
out = {}

raw = pd.read_csv(ROOT / "data" / "titanic_raw.csv")
raw["has_deck"] = raw["deck"].notna().astype(int)
y = raw["survived"]
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
tuned = dict(n_estimators=300, max_depth=6, max_features=0.5, random_state=SEED)


def score(pipe, X):
    r = cross_validate(pipe, X, y, cv=cv, scoring=["accuracy", "roc_auc"])
    return round(r["test_accuracy"].mean(), 3), round(r["test_roc_auc"].mean(), 3)


class SexOnly(FeatureEngineer):
    """Encodes sex but skips feature engineering and outlier capping."""
    def transform(self, X):
        X = X.copy(); X["sex"] = X["sex"].map({"male": 0, "female": 1}); return X


def pre(numeric, passthrough):
    return ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), numeric),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(drop="first", handle_unknown="ignore"))]), ["embarked"]),
        ("p", "passthrough", passthrough)])


steps = []
# A. minimal: numeric columns only, default model
Xa = raw[["pclass", "age", "sibsp", "parch", "fare"]]
a = Pipeline([("i", SimpleImputer(strategy="median")), ("m", RandomForestClassifier(random_state=SEED))])
steps.append(("A. Numeric columns only (pclass, age, sibsp, parch, fare)", *score(a, Xa)))
# B. + encoded sex and embarked
Xb = raw[["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]
b = Pipeline([("f", SexOnly()), ("p", pre(["age", "fare", "sibsp", "parch"], ["pclass", "sex"])),
              ("m", RandomForestClassifier(random_state=SEED))])
steps.append(("B. + encode sex (label) and embarked (one-hot)", *score(b, Xb)))
# C. + has_deck flag
Xc = raw[RAW_FEATURES]
c = Pipeline([("f", SexOnly()), ("p", pre(["age", "fare", "sibsp", "parch"], ["pclass", "sex", "has_deck"])),
              ("m", RandomForestClassifier(random_state=SEED))])
steps.append(("C. + has_deck flag (from 77%-missing deck)", *score(c, Xc)))
# D. + feature engineering & outlier capping (full FeatureEngineer), default model
d = build_pipeline(); d.set_params(model=RandomForestClassifier(random_state=SEED))
steps.append(("D. + family_size, fare_per_person, fare capping", *score(d, Xc)))
# E. + tuned hyperparameters (final deployed pipeline)
steps.append(("E. + tuned hyperparameters (final pipeline)", *score(build_pipeline(), Xc)))
out["ablation"] = [{"step": s, "cv_accuracy": acc, "cv_roc_auc": auc} for s, acc, auc in steps]

# 2. example predictions
model = joblib.load(HERE / "model" / "titanic_pipeline.joblib")
examples = pd.DataFrame([
    dict(desc="1st-class woman, 38, cabin known", pclass=1, sex="female", age=38, sibsp=1, parch=0, fare=71.28, embarked="C", has_deck=1),
    dict(desc="1st-class man, 54, cabin known", pclass=1, sex="male", age=54, sibsp=0, parch=0, fare=51.86, embarked="S", has_deck=1),
    dict(desc="2nd-class girl, 4, with parents", pclass=2, sex="female", age=4, sibsp=1, parch=1, fare=26.0, embarked="S", has_deck=0),
    dict(desc="3rd-class woman, 27, with 2 kids", pclass=3, sex="female", age=27, sibsp=0, parch=2, fare=11.13, embarked="S", has_deck=0),
    dict(desc="3rd-class man, 22, travelling alone", pclass=3, sex="male", age=22, sibsp=0, parch=0, fare=7.25, embarked="S", has_deck=0),
])
examples["survival_probability"] = model.predict_proba(examples[RAW_FEATURES])[:, 1].round(3)
out["examples"] = examples.to_dict(orient="records")

# 3. deployment performance
path = HERE / "model" / "titanic_pipeline.joblib"
t = time.perf_counter(); joblib.load(path); load_ms = (time.perf_counter() - t) * 1000
from app import app
client = app.test_client()
body = {k: v for k, v in examples.iloc[4].items() if k in RAW_FEATURES}
body = {k: (v.item() if hasattr(v, "item") else v) for k, v in body.items()}
client.post("/predict", json=body)  # warm-up
lat = []
for _ in range(50):
    t = time.perf_counter(); client.post("/predict", json=body); lat.append((time.perf_counter() - t) * 1000)
batch = [body] * 100
t = time.perf_counter(); client.post("/predict", json=batch); batch_ms = (time.perf_counter() - t) * 1000
out["performance"] = {"model_size_mb": round(path.stat().st_size / 1e6, 2), "load_ms": round(load_ms),
                      "single_median_ms": round(float(np.median(lat)), 1),
                      "batch100_ms": round(batch_ms, 1)}

# 4. validation examples
bad = client.post("/predict", json={**body, "pclass": 5, "sex": "robot"})
out["validation_example"] = {"status": bad.status_code, "body": bad.json}

(ROOT / "outputs" / "week4_impact.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
