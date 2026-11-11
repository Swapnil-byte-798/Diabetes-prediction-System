"""Train and replay a linear SVC on the repository's diabetes benchmark."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATA = Path(__file__).with_name("diabetes.csv")
FEATURES = ("Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
            "BMI", "DiabetesPedigreeFunction", "Age")
ZERO_IS_MISSING = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")
OUTCOME = "Outcome"
SEED = 2


def clean_features(frame):
    features = frame.loc[:, list(FEATURES)].apply(pd.to_numeric, errors="raise").astype(float)
    values = features.to_numpy()
    if np.isinf(values).any() or (values < 0).any():
        raise ValueError("features must be finite and nonnegative")
    required = features.drop(columns=list(ZERO_IS_MISSING))
    if required.isna().any().any() or (features["Age"] <= 0).any():
        raise ValueError("Age must be positive; Age, Pregnancies and DiabetesPedigreeFunction are required")
    measurements = features.loc[:, list(ZERO_IS_MISSING)].replace(0, np.nan)
    if measurements.isna().all(axis=1).any():
        raise ValueError("at least one measurement must be present")
    return features.assign(**{name: measurements[name] for name in ZERO_IS_MISSING})


def read_benchmark(data_path):
    data = Path(data_path).read_bytes()
    frame = pd.read_csv(BytesIO(data))
    expected = set(FEATURES) | {OUTCOME}
    if set(frame.columns) != expected:
        raise ValueError("CSV must contain the eight named features and Outcome")
    y = frame[OUTCOME]
    if not y.isin([0, 1]).all() or y.value_counts().reindex([0, 1], fill_value=0).min() < 7:
        raise ValueError("Outcome must contain at least seven rows per binary class (0 and 1)")
    return clean_features(frame), y.astype(int), sha256(data).hexdigest()


def fit_and_evaluate(data_path=DATA):
    X, y, digest = read_benchmark(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED,
    )
    pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVC(kernel="linear"))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_scores = cross_validate(
        pipeline, X_train, y_train, cv=folds,
        scoring=("accuracy", "balanced_accuracy", "recall", "roc_auc"), error_score="raise",
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    report = {
        "dataset_sha256": digest,
        "split_seed": SEED,
        "training_rows": len(y_train),
        "test_rows": len(y_test),
        "missing_measurements": {name: int(X[name].isna().sum()) for name in ZERO_IS_MISSING},
        "training_cv": {
            metric: {"mean": float(scores.mean()), "std": float(scores.std()), "folds": scores.tolist()}
            for metric, scores in fold_scores.items() if metric.startswith("test_")
        },
        "holdout": {
            "accuracy": accuracy_score(y_test, predictions),
            "majority_baseline_accuracy": float((y_test == y_train.mode().iloc[0]).mean()),
            "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, pipeline.decision_function(X_test)),
            "confusion_matrix_labels_0_1": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        },
    }
    return pipeline, report


def predict(pipeline, record):
    if not isinstance(record, dict):
        raise ValueError("input must be a JSON object with named features")
    missing = sorted(set(FEATURES) - record.keys())
    extra = sorted(record.keys() - set(FEATURES))
    if missing or extra:
        raise ValueError(f"unexpected feature set (missing: {missing}; extra: {extra})")
    for name, value in record.items():
        if value is None and name in ZERO_IS_MISSING:
            continue
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    row = clean_features(pd.DataFrame([record], columns=FEATURES))
    return {
        "predicted_class": int(pipeline.predict(row)[0]),
        "decision_score": float(pipeline.decision_function(row)[0]),
    }


def unique_fields(pairs):
    fields = dict(pairs)
    if len(fields) != len(pairs):
        raise ValueError("duplicate JSON field")
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="fit and evaluate the benchmark")
    train.add_argument("--data", type=Path, default=DATA)
    inference = commands.add_parser("predict", help="score one named JSON record")
    inference.add_argument("input", type=Path)
    for command in (train, inference):
        command.add_argument("--model", type=Path, default=Path("artifacts/diabetes.joblib"))
    args = parser.parse_args()

    try:
        if args.command == "train":
            pipeline, report = fit_and_evaluate(args.data)
            args.model.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=args.model.parent) as stage:
                pending = Path(stage) / "diabetes.joblib"
                joblib.dump({"pipeline": pipeline, "dataset_sha256": report["dataset_sha256"]}, pending)
                os.replace(pending, args.model)
            result = report
        else:
            record = json.loads(args.input.read_text(encoding="utf-8"), object_pairs_hook=unique_fields)
            bundle = joblib.load(args.model)
            result = predict(bundle["pipeline"], record)
    except (ValueError, OverflowError, OSError, EOFError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
