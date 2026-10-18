# AI Pioneers Internship — Machine Learning Projects (Weeks 1–4)

Four-week machine learning internship work by **Gurram Venkata Sai Aasrith** (YuvaIntern · AI Pioneers).
Each week builds on the previous one, ending in a deployed ML web application.

| Week | Topic | Folder | Key result |
|---|---|---|---|
| 1 | Python for ML & data preprocessing | `week1_preprocessing/` | 891×15 raw Titanic data → 891×11 clean, model-ready dataset, 0 missing values |
| 2 | Supervised learning models | `week2_supervised/` | 4 classifiers + 4 regressors compared; best F1 0.739 (KNN), best R² 0.453 (Linear Regression) |
| 3 | Unsupervised learning & model evaluation | `week3_unsupervised/` | K-Means recovers wine cultivars (ARI 0.897); tuned Random Forest CV ROC-AUC 0.891 |
| 4 | Deployment & capstone | `week4_deployment/` | Flask web app + JSON API serving a joblib pipeline, 7 automated tests passing |

## Quick start

```bash
git clone https://github.com/Aasrith1319/ai-pioneers-internship.git
cd ai-pioneers-internship
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python week1_preprocessing/preprocess.py          # writes data/titanic_clean.csv
python week2_supervised/supervised_models.py
python week3_unsupervised/clustering_evaluation.py

cd week4_deployment
python train.py                                   # writes model/titanic_pipeline.joblib
python -m pytest -q                               # 7 tests
python app.py                                     # http://127.0.0.1:5000
```

Every script saves its charts to `outputs/figures/` and its metrics to `outputs/*.json`.

## Week 1 — Data preprocessing
Dataset: Titanic passengers (891 rows). Steps: EDA, missing-value imputation (age by class × sex median, embarked by mode),
removal of leaky/redundant columns, feature engineering (`family_size`, `fare_per_person`, `has_deck`), IQR outlier capping,
label + one-hot encoding, standardization, and mutual-information feature selection.

## Week 2 — Supervised models
- **Classification** (Titanic survival): Logistic Regression, Decision Tree, Random Forest, KNN — accuracy, precision, recall, F1.
- **Regression** (scikit-learn Diabetes dataset): Linear Regression, Decision Tree, Random Forest, KNN — MAE, RMSE, R².

## Week 3 — Unsupervised learning & evaluation
- K-Means (k chosen with elbow + silhouette) and Agglomerative clustering on the Wine dataset, PCA, dendrogram.
- Stratified 5-fold cross-validation, confusion matrix, classification report, ROC-AUC, GridSearchCV (48 combinations).

## Week 4 — Deployment
`train.py` wraps all preprocessing and the tuned model in one scikit-learn `Pipeline`, serialized with joblib.
`app.py` serves it with Flask:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" \
  -d '{"pclass":3,"sex":"male","age":22,"sibsp":1,"parch":0,"fare":7.25,"embarked":"S","has_deck":0}'
# {"survival_probability": 0.1196, "survived": 0}
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET/POST | HTML prediction form |
| `/predict` | POST | JSON prediction (single object or list of up to 1000), with input validation |
| `/health` | GET | Status and hold-out metrics |

## Project structure
```
data/                 raw and cleaned datasets
week1_preprocessing/  preprocess.py
week2_supervised/     supervised_models.py
week3_unsupervised/   clustering_evaluation.py
week4_deployment/     train.py, features.py, app.py, templates/, test_app.py, model/
outputs/              figures/ and metrics JSON for every week
```
