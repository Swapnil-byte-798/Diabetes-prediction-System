import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from diabetes import DATA, FEATURES, SEED, fit_and_evaluate, predict, read_benchmark


ROOT = DATA.parent
EXAMPLE = json.loads((ROOT / "examples/record.json").read_text())


@pytest.fixture(scope="module")
def fitted():
    return fit_and_evaluate()


def test_original_csv_and_training_only_preprocessing(fitted):
    pipeline, report = fitted
    X, y, digest = read_benchmark(DATA)
    assert digest == "698c203a14aa31941d2251175330c9199f3ccdb31597abbba2a3e35416257a72"
    assert y.value_counts().to_dict() == {0: 500, 1: 268}
    assert report["missing_measurements"] == {
        "Glucose": 5, "BloodPressure": 35, "SkinThickness": 227, "Insulin": 374, "BMI": 11,
    }
    assert X["Pregnancies"].eq(0).any()
    assert X["Pregnancies"].isna().sum() == 0

    training, holdout, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    assert (len(training), len(holdout)) == (614, 154)
    assert not training.index.intersection(holdout.index).size
    imputer = pipeline.named_steps["simpleimputer"]
    scaler = pipeline.named_steps["standardscaler"]
    np.testing.assert_allclose(imputer.statistics_, training.median().to_numpy())
    np.testing.assert_allclose(scaler.mean_, imputer.transform(training).mean(axis=0))
    assert scaler.n_samples_seen_ == len(training)
    assert all(len(metric["folds"]) == 5 for metric in report["training_cv"].values())
    assert report["holdout"]["confusion_matrix_labels_0_1"] == [[91, 9], [26, 28]]
    assert report["holdout"]["roc_auc"] == pytest.approx(0.82)


def test_named_features_and_missing_measurements(fitted):
    model, _ = fitted
    prediction = predict(model, EXAMPLE)
    assert prediction["predicted_class"] == 0
    assert predict(model, dict(reversed(tuple(EXAMPLE.items())))) == prediction
    assert predict(model, {**EXAMPLE, "Insulin": 0}) == predict(model, {**EXAMPLE, "Insulin": None})


@pytest.mark.parametrize("invalid", [
    [2, 110, 74, 29, 125, 32.4, 0.698, 27],
    {key: value for key, value in EXAMPLE.items() if key != "Glucose"},
    {**EXAMPLE, "Outcome": 0},
    {**EXAMPLE, "Glucose": "110"},
    {**EXAMPLE, "Glucose": float("nan")},
    {**EXAMPLE, "Glucose": float("inf")},
    {**EXAMPLE, "Glucose": -1},
    {**EXAMPLE, "BMI": -1},
    {**EXAMPLE, "Age": -1},
    {**EXAMPLE, "Age": 0},
    {**EXAMPLE, "Pregnancies": -1},
    {**EXAMPLE, "Pregnancies": True},
    {**EXAMPLE, "Age": None},
    dict.fromkeys(FEATURES, 0),
    {**EXAMPLE, "Glucose": 0, "BloodPressure": 0, "SkinThickness": 0, "Insulin": 0, "BMI": 0},
])
def test_invalid_inputs_are_rejected(fitted, invalid):
    with pytest.raises(ValueError):
        predict(fitted[0], invalid)


def test_invalid_csv_schema(tmp_path):
    invalid = tmp_path / "invalid.csv"
    invalid.write_text("Pregnancies,Outcome\n2,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="eight named features"):
        read_benchmark(invalid)


def test_cli_trains_and_reloads_from_fresh_directory(tmp_path, fitted):
    command = [sys.executable, str(ROOT / "diabetes.py")]

    def run(*args):
        return subprocess.run([*command, *args], cwd=tmp_path, text=True, capture_output=True)

    training = run("train")
    assert training.returncode == 0, training.stderr
    assert json.loads(training.stdout) == fitted[1]
    assert (tmp_path / "artifacts/diabetes.joblib").is_file()
    scoring = run("predict", str(ROOT / "examples/record.json"))
    assert scoring.returncode == 0, scoring.stderr
    assert json.loads(scoring.stdout) == predict(fitted[0], EXAMPLE)

    invalid = tmp_path / "duplicate.json"
    invalid.write_text('{"Glucose": 100, "Glucose": 110}', encoding="utf-8")
    failure = run("predict", str(invalid))
    assert failure.returncode == 2
    assert failure.stdout == "" and "duplicate JSON field" in failure.stderr
    missing = run("predict", str(ROOT / "examples/record.json"), "--model", "missing.joblib")
    assert missing.returncode == 2 and "missing.joblib" in missing.stderr
