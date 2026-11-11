# Diabetes Prediction System execution audit

Updated 11 November 2026. The repaired CLI and notebook run from a fresh checkout. The linear SVC still misses 26 of 54 positive records on the historical holdout.

## What failed and what changed

The baseline checkout was `3773acfdfee6e2ddd765461559d3e95820281fb8`. Running the original notebook cell produced FileNotFoundError because it asked for diabetes.csv and the repository had diabetes.xls. Copying the same bytes to the expected name exposed a second error under pandas 3.1.0 because the drop call supplied both columns and axis. The unchanged cell ran under pandas 2.2.3 with the temporary alias.

The repair renames the CSV without changing its bytes, selects features by name and pins the dependencies. The CLI and notebook use the same fit and predict functions. The pipeline fits median imputation, standardisation and a linear SVC, then saves them together as a joblib artifact. The separate plant notebook was moved to archive/ with identical bytes. Its private data dependency and reused validation generator remain outside the diabetes run.

## What actually ran

| Check | Observed result |
|---|---|
| Clean clone of tested GitHub commit | Trained from the bundled CSV, saved the pipeline, then scored the supplied JSON in a separate process |
| pandas 2.2.3 and 3.1.0 | 19 tests passed per version locally. Both GitHub Actions jobs passed |
| Real notebook kernel | Executed successfully in both GitHub Actions jobs. The local workspace blocked ZeroMQ socket setup |
| Training-only preprocessing | Split indices disjoint, imputation medians matched the training data, scaler fitted to 614 imputed training rows |
| Named input | Original and reversed key order yielded the same class and score. Null insulin and coded zero insulin were equivalent |
| Malformed input | Missing and extra keys, duplicate keys, positional records, strings, non-finite values, invalid negatives and all-missing measurements were rejected |
| Saved model | A new CLI process loaded the fitted bundle. A missing model path returned an error |
| Sample output | Class 0 and a decision score of approximately -1.1125 |

The runtime commit tested by these jobs was `0eaf36a4fb27fd225d3cee0e00eb88617fda9c38`. The jobs installed their dependencies, ran the tests, trained and reloaded the model, then executed the notebook with a Python kernel. [Passing workflow](https://github.com/Swapnil-byte-798/Diabetes-prediction-System/actions/runs/43079099911)

## Data and split

The CSV SHA-256 is `698c203a14aa31941d2251175330c9199f3ccdb31597abbba2a3e35416257a72`. There are 768 records with eight features and the Outcome label, including 500 negatives and 268 positives. The original stratified split with seed 2 was retained, with 614 training and 154 holdout records. The CSV contains 5 zero-coded glucose readings, 35 blood pressure readings, 227 skin thickness readings, 374 insulin readings and 11 BMI readings. The loader maps these five measurement codes to missing values. Zero pregnancies remains valid. [Dataset reference](https://search.r-project.org/CRAN/refmans/mlbench/html/PimaIndiansDiabetes.html)

## Historical split comparison

| Metric | Original SVC | Repaired SVC |
|---|---:|---:|
| Accuracy | 77.27% (119/154) | 77.27% (119/154) |
| Positive precision | 75.68% | 75.68% |
| Positive recall | 51.85% (28/54) | 51.85% (28/54) |
| Balanced accuracy | 71.43% | 71.43% |
| ROC AUC | 0.7919 | 0.8200 |
| False negatives | 26 | 26 |

Both confusion matrices are `[[91, 9], [26, 28]]`, with actual labels as rows and predicted labels as columns in class order 0, 1. The majority baseline scores 64.94%. Improved AUC reflects the ranking of continuous scores. The default decision rule still yields the same totals. The earlier audit had already examined the holdout, so the comparison is a replay on the same records.

Five-fold stratified validation on the 614 training records averaged 77.04% accuracy, 72.17% balanced accuracy, 56.09% positive recall and 0.8451 ROC AUC. Preprocessing and training were repeated inside each fold. These are small benchmark estimates. External clinical validity and individual risk calibration remain untested.
