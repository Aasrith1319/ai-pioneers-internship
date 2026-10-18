"""
Week 1 - Python for Machine Learning & Data Preprocessing
Dataset: Titanic passenger data (891 rows, 15 columns)

Steps:
  1. Load data and inspect structure
  2. Exploratory data analysis (EDA)
  3. Handle missing values
  4. Remove redundant / leaky columns
  5. Feature engineering
  6. Encode categorical variables
  7. Handle outliers
  8. Normalize numeric features
  9. Feature selection
 10. Save the cleaned dataset
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
log = {}

# ---------------------------------------------------------------- 1. Load
df = pd.read_csv(DATA / "titanic_raw.csv")
print("Raw shape:", df.shape)
print(df.dtypes)
log["raw_shape"] = list(df.shape)
log["duplicates"] = int(df.duplicated().sum())

# ---------------------------------------------------------------- 2. EDA
missing = df.isna().sum().sort_values(ascending=False)
log["missing_before"] = {k: int(v) for k, v in missing[missing > 0].items()}
print("\nMissing values:\n", missing[missing > 0])
print("\nSummary statistics:\n", df.describe().T)
log["survival_rate"] = round(float(df["survived"].mean()), 3)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.countplot(data=df, x="sex", hue="survived", ax=axes[0])
axes[0].set_title("Survival by sex")
sns.countplot(data=df, x="pclass", hue="survived", ax=axes[1])
axes[1].set_title("Survival by passenger class")
sns.histplot(data=df, x="age", hue="survived", bins=30, kde=True, ax=axes[2])
axes[2].set_title("Age distribution by survival")
plt.tight_layout(); plt.savefig(FIG / "w1_eda_overview.png", dpi=130); plt.close()

plt.figure(figsize=(8, 4))
(df.isna().mean() * 100).sort_values(ascending=False).plot.bar(color="#d9534f")
plt.ylabel("% missing"); plt.title("Missing values per column (raw data)")
plt.tight_layout(); plt.savefig(FIG / "w1_missing.png", dpi=130); plt.close()

# ------------------------------------------------- 3. Remove redundant columns
# 'alive' is the target in text form (leakage); class/embark_town/who/adult_male
# duplicate information already present in pclass/embarked/sex+age.
# 'deck' is 77% missing, so it is reduced to a binary "cabin known" flag.
df["has_deck"] = df["deck"].notna().astype(int)
df = df.drop(columns=["alive", "class", "embark_town", "who", "adult_male", "deck"])
# Note: identical rows exist because the dataset has no name/ticket ID; they are
# different passengers, so they are kept (dropping them would bias the data).
log["after_drop_shape"] = list(df.shape)

# ------------------------------------------------- 4. Missing values
# Age: median within (pclass, sex) groups - more accurate than a global median
df["age"] = df.groupby(["pclass", "sex"])["age"].transform(lambda s: s.fillna(s.median()))
# Embarked: only 2 missing -> mode
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
log["missing_after"] = int(df.isna().sum().sum())

# ------------------------------------------------- 5. Feature engineering
df["family_size"] = df["sibsp"] + df["parch"] + 1
df["alone"] = df["alone"].astype(int)
df["fare_per_person"] = df["fare"] / df["family_size"]
df["age_group"] = pd.cut(df["age"], bins=[0, 12, 18, 35, 60, 100],
                         labels=["child", "teen", "young_adult", "adult", "senior"])

# ------------------------------------------------- 6. Outliers (IQR capping on fare)
for col in ["fare", "fare_per_person"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    log[f"{col}_outliers_capped"] = int((df[col] > upper).sum())
    df[col] = df[col].clip(upper=upper)

# ------------------------------------------------- 7. Encoding
df["sex"] = df["sex"].map({"male": 0, "female": 1})               # binary -> label
df = pd.get_dummies(df, columns=["embarked", "age_group"], drop_first=True, dtype=int)  # nominal -> one-hot

# ------------------------------------------------- 8. Normalization
num_cols = ["age", "fare", "fare_per_person", "sibsp", "parch", "family_size"]
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[num_cols] = scaler.fit_transform(df[num_cols])

# ------------------------------------------------- 9. Feature selection
X = df_scaled.drop(columns="survived"); y = df_scaled["survived"]
mi = pd.Series(mutual_info_classif(X, y, random_state=42), index=X.columns).sort_values(ascending=False)
log["mutual_information"] = {k: round(float(v), 4) for k, v in mi.items()}

plt.figure(figsize=(8, 5))
mi.plot.barh(color="#2c7fb8"); plt.gca().invert_yaxis()
plt.title("Feature importance (mutual information with survival)")
plt.tight_layout(); plt.savefig(FIG / "w1_feature_selection.png", dpi=130); plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(df_scaled.corr(), cmap="coolwarm", center=0, annot=False)
plt.title("Correlation matrix after preprocessing")
plt.tight_layout(); plt.savefig(FIG / "w1_correlation.png", dpi=130); plt.close()

# Drop redundant features: sibsp and alone are fully captured by family_size,
# and age_group dummies duplicate the continuous age column (kept for EDA only).
low_info = ["sibsp", "alone"] + [c for c in df_scaled.columns if c.startswith("age_group_")]
df_scaled = df_scaled.drop(columns=low_info)
log["dropped_low_information"] = low_info

# ------------------------------------------------- 10. Save
df_scaled.to_csv(DATA / "titanic_clean.csv", index=False)
log["final_shape"] = list(df_scaled.shape)
log["final_columns"] = df_scaled.columns.tolist()
(ROOT / "outputs" / "week1_log.json").write_text(json.dumps(log, indent=2))
print("\nClean shape:", df_scaled.shape)
print(json.dumps(log, indent=2))
