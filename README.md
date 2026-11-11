# Diabetes Prediction System

A reproducible Python benchmark for binary classification on the bundled 768-row Pima diabetes dataset. The model uses median imputation, standardisation and a linear support vector classifier. The command line and notebook share the same implementation.

## Run from a fresh checkout

Use Python 3.12. In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python diabetes.py train
python diabetes.py predict examples/record.json
```

Training prints the dataset hash, fold metrics and holdout metrics as JSON and saves the fitted imputer, scaler and classifier in `artifacts/diabetes.joblib`. Training again replaces the artifact at the selected path. Use `--model` to retain a separate run, and load model files from a trusted source.

Prediction accepts one JSON object with the eight names shown in the example. Use `null` or the dataset's zero code for a missing Glucose, BloodPressure, SkinThickness, Insulin or BMI measurement. Supply at least one of those measurements. A pregnancy count of zero is valid. Class 1 denotes the positive benchmark label, and the decision score is the signed SVC margin.

To run the tests and execute the notebook from the repository root:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
jupyter execute --timeout=60 Diabetes.ipynb
```

## Verified results

| Metric | 154-record holdout |
|---|---:|
| Accuracy | 77.27% (119/154) |
| Majority baseline accuracy | 64.94% (100/154) |
| Balanced accuracy | 71.43% |
| Positive recall | 51.85% (28/54) |
| ROC AUC | 0.8200 |

The original split is preserved for comparison. The repaired pipeline missed 26 of 54 positives. Five-fold cross-validation within the 614 training rows averaged 77.04% accuracy and 56.09% positive recall. These are benchmark results. External clinical validity and individual risk calibration have not been established.

The filename was corrected with identical data bytes. The [dataset reference](https://search.r-project.org/CRAN/refmans/mlbench/html/PimaIndiansDiabetes.html) documents the five zero-coded measurements. The [audit](docs/verification.md) records the hash and execution results, and the [study material](docs/README.md) explains the workflow and interview questions.

The separate plant-image notebook is preserved in `archive/`. It depends on private Drive data and has its own unresolved validation split.
