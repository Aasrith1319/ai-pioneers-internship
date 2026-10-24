"""
Week 3 - Unsupervised Learning & Model Evaluation

Part A - Clustering on the Wine dataset (178 wines, 13 chemical measurements)
         K-Means (elbow + silhouette to choose k), Agglomerative (hierarchical)
         clustering with a dendrogram, PCA for dimensionality reduction.
Part B - Model evaluation on the Titanic classifier from Week 2
         5-fold cross-validation, confusion matrix, precision / recall / F1,
         ROC-AUC, and hyperparameter tuning with GridSearchCV.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, adjusted_rand_score, davies_bouldin_score,
                             confusion_matrix, classification_report, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
SEED = 42
out = {}

# =================================================================== A. Clustering
wine = load_wine(as_frame=True)
Xw = StandardScaler().fit_transform(wine.data)          # scaling is essential for distance-based clustering
true = wine.target

# --- choose k
ks = range(2, 11); inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xw)
    inertia.append(km.inertia_); sil.append(silhouette_score(Xw, km.labels_))
best_k = list(ks)[int(np.argmax(sil))]
out["k_selection"] = {"k": list(ks), "inertia": [round(i, 1) for i in inertia], "silhouette": [round(s, 3) for s in sil], "best_k": best_k}

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow method"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia (WCSS)")
ax[1].plot(ks, sil, "o-", color="green"); ax[1].axvline(best_k, ls="--", c="r"); ax[1].set_title("Silhouette score vs k"); ax[1].set_xlabel("k")
plt.tight_layout(); plt.savefig(FIG / "w3_choose_k.png", dpi=130); plt.close()

# --- fit both algorithms
km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(Xw)
agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit(Xw)
clust = {}
for name, labels in [("K-Means", km.labels_), ("Agglomerative (Ward)", agg.labels_)]:
    clust[name] = {
        "silhouette": round(silhouette_score(Xw, labels), 3),
        "davies_bouldin": round(davies_bouldin_score(Xw, labels), 3),
        "adjusted_rand_vs_true_cultivar": round(adjusted_rand_score(true, labels), 3),
    }
out["clustering"] = clust
print(pd.DataFrame(clust).T)

# --- PCA
pca = PCA().fit(Xw)
cum = np.cumsum(pca.explained_variance_ratio_)
out["pca"] = {"pc1_pc2_variance": round(float(cum[1]), 3),
              "components_for_90pct": int(np.argmax(cum >= 0.90) + 1)}
Z = PCA(n_components=2).fit_transform(Xw)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].plot(range(1, 14), cum, "o-"); ax[0].axhline(.9, ls="--", c="r")
ax[0].set_title("PCA cumulative explained variance"); ax[0].set_xlabel("Components")
for a, labels, title in [(ax[1], km.labels_, "K-Means clusters (PCA view)"),
                         (ax[2], true, "True cultivar labels (PCA view)")]:
    a.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="viridis", s=25)
    a.set_title(title); a.set_xlabel("PC1"); a.set_ylabel("PC2")
cent = PCA(n_components=2).fit(Xw).transform(km.cluster_centers_)
ax[1].scatter(cent[:, 0], cent[:, 1], c="red", marker="X", s=200)
plt.tight_layout(); plt.savefig(FIG / "w3_pca_clusters.png", dpi=130); plt.close()

plt.figure(figsize=(11, 4.5))
dendrogram(linkage(Xw, "ward"), truncate_mode="lastp", p=30, leaf_rotation=90)
plt.title("Hierarchical clustering dendrogram (Ward linkage)"); plt.ylabel("Distance")
plt.tight_layout(); plt.savefig(FIG / "w3_dendrogram.png", dpi=130); plt.close()

profile = wine.data.assign(cluster=km.labels_).groupby("cluster").mean().round(2)
out["cluster_profile"] = profile[["alcohol", "malic_acid", "flavanoids", "color_intensity", "proline"]].to_dict()

# =================================================================== B. Model evaluation
df = pd.read_csv(ROOT / "data" / "titanic_clean.csv")
X, y = df.drop(columns="survived"), df["survived"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest (default)": RandomForestClassifier(random_state=SEED),
}
cv_rows = {}
for name, m in models.items():
    s = cross_validate(m, X_tr, y_tr, cv=cv, scoring=scoring)
    cv_rows[name] = {k: f"{s['test_'+k].mean():.3f} ± {s['test_'+k].std():.3f}" for k in scoring}

grid = GridSearchCV(RandomForestClassifier(random_state=SEED),
                    {"n_estimators": [100, 300], "max_depth": [4, 6, 8, None],
                     "min_samples_leaf": [1, 3, 5], "max_features": ["sqrt", 0.5]},
                    cv=cv, scoring="roc_auc", n_jobs=-1).fit(X_tr, y_tr)
out["grid_search"] = {"best_params": grid.best_params_, "best_cv_roc_auc": round(grid.best_score_, 3),
                      "combinations_tested": len(grid.cv_results_["params"])}
s = cross_validate(grid.best_estimator_, X_tr, y_tr, cv=cv, scoring=scoring)
cv_rows["Random Forest (tuned)"] = {k: f"{s['test_'+k].mean():.3f} ± {s['test_'+k].std():.3f}" for k in scoring}
out["cross_validation"] = cv_rows
print(pd.DataFrame(cv_rows).T)

final = {"Logistic Regression": models["Logistic Regression"].fit(X_tr, y_tr),
         "Random Forest (default)": models["Random Forest (default)"].fit(X_tr, y_tr),
         "Random Forest (tuned)": grid.best_estimator_}
test = {}
plt.figure(figsize=(6, 5))
for name, m in final.items():
    proba = m.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, proba)
    auc = roc_auc_score(y_te, proba); test[name] = {"test_roc_auc": round(auc, 3)}
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves - held-out test set"); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig(FIG / "w3_roc.png", dpi=130); plt.close()

best = grid.best_estimator_; pred = best.predict(X_te)
cm = confusion_matrix(y_te, pred)
plt.figure(figsize=(4.5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix - tuned Random Forest")
plt.tight_layout(); plt.savefig(FIG / "w3_confusion.png", dpi=130); plt.close()

rep = classification_report(y_te, pred, target_names=["Died", "Survived"], output_dict=True)
out["test_set"] = test
out["tuned_rf_report"] = {k: {m: round(v, 3) for m, v in d.items()} if isinstance(d, dict) else round(d, 3) for k, d in rep.items()}
out["confusion_matrix"] = cm.tolist()
print(classification_report(y_te, pred, target_names=["Died", "Survived"]))
(ROOT / "outputs" / "week3_results.json").write_text(json.dumps(out, indent=2, default=str))
print(json.dumps({k: out[k] for k in ["k_selection", "pca", "grid_search", "test_set"]}, indent=1, default=str))
