"""
Week 4 - Flask deployment of the Titanic survival model.

Endpoints
  GET  /          HTML form for interactive predictions
  GET  /health    liveness check + model metadata
  POST /predict   JSON API: one passenger object or a list of them

Run:   python app.py      (then open http://127.0.0.1:5000)
"""
from pathlib import Path
import json

import joblib
import pandas as pd
from flask import Flask, jsonify, request, render_template

from features import RAW_FEATURES  # noqa: F401  (needed so joblib can unpickle FeatureEngineer)

HERE = Path(__file__).resolve().parent
MODEL = joblib.load(HERE / "model" / "titanic_pipeline.joblib")
META = json.loads((HERE / "model" / "metadata.json").read_text())

SCHEMA = {
    "pclass": (int, {1, 2, 3}),
    "sex": (str, {"male", "female"}),
    "age": (float, (0, 100)),
    "sibsp": (int, (0, 10)),
    "parch": (int, (0, 10)),
    "fare": (float, (0, 600)),
    "embarked": (str, {"S", "C", "Q"}),
    "has_deck": (int, {0, 1}),
}

app = Flask(__name__)


def validate(record):
    """Return (clean_record, errors). Converts types and checks allowed values/ranges."""
    clean, errors = {}, []
    for field, (typ, allowed) in SCHEMA.items():
        if field not in record or record[field] in ("", None):
            errors.append(f"'{field}' is required"); continue
        try:
            val = typ(record[field])
            if typ is str:
                val = val.strip().lower() if field == "sex" else val.strip().upper()
        except (TypeError, ValueError):
            errors.append(f"'{field}' must be of type {typ.__name__}"); continue
        if isinstance(allowed, set) and val not in allowed:
            errors.append(f"'{field}' must be one of {sorted(allowed)}")
        elif isinstance(allowed, tuple) and not (allowed[0] <= val <= allowed[1]):
            errors.append(f"'{field}' must be between {allowed[0]} and {allowed[1]}")
        clean[field] = val
    return clean, errors


def predict_records(records):
    df = pd.DataFrame(records)[list(SCHEMA)]
    proba = MODEL.predict_proba(df)[:, 1]
    return [{"survived": int(p >= 0.5), "survival_probability": round(float(p), 4)} for p in proba]


@app.get("/health")
def health():
    return jsonify(status="ok", model=META["model"], holdout_metrics=META["holdout_metrics"])


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify(error="Request body must be JSON"), 400
    records = payload if isinstance(payload, list) else [payload]
    if not records or len(records) > 1000:
        return jsonify(error="Send between 1 and 1000 records"), 400
    clean, all_errors = [], {}
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            all_errors[i] = ["record must be a JSON object"]; continue
        c, errs = validate(rec)
        if errs:
            all_errors[i] = errs
        clean.append(c)
    if all_errors:
        return jsonify(error="Validation failed", details=all_errors), 422
    results = predict_records(clean)
    return jsonify(results if isinstance(payload, list) else results[0])


@app.route("/", methods=["GET", "POST"])
def index():
    result, errors, form = None, [], {}
    if request.method == "POST":
        form = request.form.to_dict()
        form.setdefault("has_deck", "0")
        clean, errors = validate(form)
        if not errors:
            result = predict_records([clean])[0]
    return render_template("index.html", result=result, errors=errors, form=form, meta=META)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
