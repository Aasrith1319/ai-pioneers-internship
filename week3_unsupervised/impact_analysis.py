"""
Week 3 - Impact analysis: measures how much each step changes the result.

  1. Feature scaling        -> clustering quality with vs without StandardScaler
  2. PCA                    -> clustering on 13 features vs 2 / 5 / 8 principal components
  3. Linkage method         -> ward vs complete vs average vs single
  4. Cross-validation       -> spread of accuracy across 5 single splits vs 5-fold CV
  5. Hyperparameter tuning  -> effect of each max_depth value on train/CV accuracy
  6. Decision threshold     -> precision/recall trade-off at different thresholds
Also prints concrete example wines and passengers used in the report.
"""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, adjusted_rand_score, precision_score,
                             recall_score, f1_score, accuracy_score)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
out = {}

wine = load_wine(as_frame=True)
raw, true = wine.data, wine.target
Xs = StandardScaler().fit_transform(raw)

def km_scores(X):
    lab = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit_predict(X)
    return lab, round(adjusted_rand_score(true, lab), 3)

# 1. scaling
lab_raw, ari_raw = km_scores(raw.values)
lab_s, ari_s = km_scores(Xs)
out["scaling"] = {"unscaled_ari": ari_raw, "scaled_ari": ari_s,
                  "proline_range": [float(raw.proline.min()), float(raw.proline.max())],
                  "hue_range": [float(raw.hue.min()), float(raw.hue.max())]}

# 2. PCA components
pca_rows = []
for n in [2, 5, 8, 13]:
    Z = PCA(n_components=n, random_state=SEED).fit_transform(Xs)
    t = time.perf_counter(); lab, ari = km_scores(Z); ms = (time.perf_counter() - t) * 1000
    pca_rows.append({"components": n, "ari": ari, "silhouette": round(silhouette_score(Z, lab), 3)})
out["pca_components"] = pca_rows

# 3. linkage
out["linkage"] = {}
for link in ["ward", "complete", "average", "single"]:
    lab = AgglomerativeClustering(n_clusters=3, linkage=link).fit_predict(Xs)
    out["linkage"][link] = {"ari": round(adjusted_rand_score(true, lab), 3),
                            "cluster_sizes": sorted(np.bincount(lab).tolist(), reverse=True)}

# example wines: one per cluster, the one closest to its centroid
km = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(Xs)
d = km.transform(Xs)
ex = []
for c in range(3):
    i = int(np.argmin(np.where(km.labels_ == c, d[:, c], np.inf)))
    r = raw.iloc[i]
    ex.append({"wine_index": i, "cluster": c, "true_cultivar": int(true[i]),
               "alcohol": r.alcohol, "flavanoids": r.flavanoids,
               "color_intensity": r.color_intensity, "proline": r.proline})
out["example_wines"] = ex
ct = pd.crosstab(km.labels_, true)
out["cluster_vs_cultivar"] = ct.values.tolist()

# ---------------- evaluation
df = pd.read_csv(ROOT / "data" / "titanic_clean.csv")
X, y = df.drop(columns="survived"), df["survived"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
rf = RandomForestClassifier(n_estimators=300, max_depth=6, max_features=0.5, random_state=SEED)

# 4. single split variability vs CV
single = []
for s in range(5):
    a, b, c, d_ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=s)
    single.append(accuracy_score(d_, rf.fit(a, c).predict(b)))
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
cvs = cross_val_score(rf, X, y, cv=cv)
out["split_variability"] = {"single_split_accs": [round(v, 3) for v in single],
                            "single_range": [round(min(single), 3), round(max(single), 3)],
                            "cv_mean": round(cvs.mean(), 3), "cv_std": round(cvs.std(), 3)}

# 5. max_depth effect
depth = []
for md in [2, 4, 6, 8, 12, None]:
    m = RandomForestClassifier(n_estimators=300, max_depth=md, max_features=0.5, random_state=SEED)
    r = cross_validate(m, X_tr, y_tr, cv=cv, scoring="accuracy", return_train_score=True)
    depth.append({"max_depth": "None" if md is None else md,
                  "train_acc": round(r["train_score"].mean(), 3), "cv_acc": round(r["test_score"].mean(), 3),
                  "gap": round(r["train_score"].mean() - r["test_score"].mean(), 3)})
out["depth_effect"] = depth

# 6. threshold
rf.fit(X_tr, y_tr); proba = rf.predict_proba(X_te)[:, 1]
th = []
for t in [0.3, 0.4, 0.5, 0.6]:
    p = (proba >= t).astype(int)
    th.append({"threshold": t, "precision": round(precision_score(y_te, p), 3),
               "recall": round(recall_score(y_te, p), 3), "f1": round(f1_score(y_te, p), 3),
               "survivors_found": int(((p == 1) & (y_te == 1)).sum()), "of": int(y_te.sum())})
out["threshold"] = th

(ROOT / "outputs" / "week3_impact.json").write_text(json.dumps(out, indent=2, default=float))
print(json.dumps(out, indent=1, default=float))
