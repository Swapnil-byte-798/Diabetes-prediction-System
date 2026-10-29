# Diabetes Prediction System

A basic tabular classification notebook using a stratified split, training-only standardisation and a linear SVC. The audit reproduced its recorded 77.27% held-out accuracy under a compatible environment and filename workaround. Positive recall was 51.85%: 26 of 54 positive test records were missed.

## Study material

The [documentation index](docs/README.md) links current Markdown and PDF architecture notes, interview questions and a tool primer. They include worked metrics, analogies, measured results and unresolved defects. This documentation update does not change the modelling code.

## Current execution status

- `Diabetes.ipynb` requests `diabetes.csv`, but the committed `diabetes.xls` is CSV text with a different filename.
- With a temporary identical-byte filename alias, pandas 3.1.0 then rejects `drop(columns='Outcome', axis=1)`.
- The unchanged code completed with Python 3.12.14, pandas 2.2.3, NumPy 2.2.6, scikit-learn 1.9.1 and SciPy 1.18.1. The repository has no declared runtime dependency manifest.
- Training-only scaling was verified; no scaling leakage was found in this notebook.
- Zero-coded missing measurements and invalid numeric inputs remain untreated. There is no saved diabetes-model bundle or serving entrypoint.

To study the compatible baseline, use an isolated environment with the recorded versions and a temporary `diabetes.csv` copy of the supplied CSV text. These are explicit workarounds, not fixes applied by this documentation commit.

## Audited result

| Actual label | Predicted 0 | Predicted 1 |
|---|---:|---:|
| 0 | 91 | 9 |
| 1 | 26 | 28 |

The split contains 614 training and 154 test rows. Accuracy is 77.27%, positive precision 75.68%, positive recall 51.85% and balanced accuracy 71.43%. Auditor-added five-fold evaluation averaged 77.73% accuracy and 57.05% recall; it is not evaluation already implemented in the notebook.

The data is a small benchmark. Its hard labels are not validated diagnoses or calibrated individual risk. Jupyter kernel sockets were blocked in the audit workspace, so the actual cell source was executed directly as Python; notebook-kernel/UI execution was not verified.

## Other notebook

`project2_capstone.ipynb` is an unrelated plant-image project. It needs private Google Drive data absent from the repository and its fit call reuses the training generator as validation. Its saved plots are not diabetes performance evidence.

## Next engineering work

Fix file/dependency setup, simplify the drop call, add named feature validation and training-only missing-data handling, persist preprocessing with the classifier, and add repeatable tests. These remain proposals, not completed features.
