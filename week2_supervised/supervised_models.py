"""
Week 2 - Supervised Machine Learning Models (scikit-learn)

Part A - Classification: predict Titanic survival (cleaned data from Week 1)
         Models: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors
Part B - Regression: predict disease progression (sklearn Diabetes dataset)
         Models: Linear Regression, Decision Tree Regressor, Random Forest Regressor, KNN Regressor
"""
from pathlib import Path
import json

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
SEED = 42
results = {}

# =========================================================== A. Classification
df = pd.read_csv(ROOT / "data" / "titanic_clean.csv")
X, y = df.drop(columns="survived"), df["survived"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"Classification: train={len(X_tr)} test={len(X_te)}")

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=7, random_state=SEED),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
}
cls_rows = []
for name, model in classifiers.items():
    model.fit(X_tr, y_tr)
    p = model.predict(X_te)
    cls_rows.append({
        "Model": name,
        "Train Acc": round(accuracy_score(y_tr, model.predict(X_tr)), 3),
        "Test Acc": round(accuracy_score(y_te, p), 3),
        "Precision": round(precision_score(y_te, p), 3),
        "Recall": round(recall_score(y_te, p), 3),
        "F1": round(f1_score(y_te, p), 3),
    })
cls_df = pd.DataFrame(cls_rows).sort_values("F1", ascending=False)
print(cls_df.to_string(index=False))
results["classification"] = cls_df.to_dict(orient="records")

ax = cls_df.set_index("Model")[["Test Acc", "Precision", "Recall", "F1"]].plot.bar(figsize=(9, 4.5), rot=0)
ax.set_ylim(0.5, 1); ax.set_title("Classification models - test-set metrics (Titanic)")
ax.legend(loc="lower right"); plt.tight_layout(); plt.savefig(FIG / "w2_classification.png", dpi=130); plt.close()

rf = classifiers["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
imp.plot.barh(figsize=(7, 4.5), color="#41ab5d"); plt.title("Random Forest feature importance")
plt.tight_layout(); plt.savefig(FIG / "w2_rf_importance.png", dpi=130); plt.close()

# =========================================================== B. Regression
diab = load_diabetes(as_frame=True)
Xr, yr = diab.data, diab.target
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=SEED)
print(f"\nRegression: train={len(Xr_tr)} test={len(Xr_te)}, features={Xr.shape[1]}")

regressors = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=4, random_state=SEED),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=6, random_state=SEED),
    "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=9)),
}
reg_rows, preds = [], {}
for name, model in regressors.items():
    model.fit(Xr_tr, yr_tr)
    p = model.predict(Xr_te); preds[name] = p
    reg_rows.append({
        "Model": name,
        "MAE": round(mean_absolute_error(yr_te, p), 2),
        "RMSE": round(mean_squared_error(yr_te, p) ** 0.5, 2),
        "R2": round(r2_score(yr_te, p), 3),
    })
reg_df = pd.DataFrame(reg_rows).sort_values("R2", ascending=False)
print(reg_df.to_string(index=False))
results["regression"] = reg_df.to_dict(orient="records")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
best = reg_df.iloc[0]["Model"]
axes[0].scatter(yr_te, preds[best], alpha=.7)
lims = [yr_te.min(), yr_te.max()]; axes[0].plot(lims, lims, "r--")
axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted"); axes[0].set_title(f"{best}: actual vs predicted")
reg_df.set_index("Model")["R2"].plot.bar(ax=axes[1], rot=15, color="#fd8d3c")
axes[1].set_title("R² score by regression model")
plt.tight_layout(); plt.savefig(FIG / "w2_regression.png", dpi=130); plt.close()

(ROOT / "outputs" / "week2_results.json").write_text(json.dumps(results, indent=2))
