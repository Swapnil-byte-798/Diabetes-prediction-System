# Diabetes Prediction System: Interview Questions

Updated 29 October 2026. Notebook code reviewed: `f2786e3`. Distinguish the notebook's implementation from audit-only evaluation and proposed repairs.

## 1. Explain the scope

### What does the project implement?

A tabular binary-classification baseline: load the supplied dataset, split rows with stratification, fit a training-only scaler, train a linear SVC and predict an example. It is not a deployed diagnostic application or calibrated risk service.

### Does it run from a fresh checkout?

Not as supplied. The notebook requests diabetes.csv, but the file is named diabetes.xls and is actually CSV text. With a temporary alias, pandas 3.1 then rejects the drop call's simultaneous columns and axis arguments. The unchanged source completed under pandas 2.2.3.

### What was actually run during the audit?

The source cell was executed as ordinary Python, first reproducing the failures and then completing with explicit workarounds. Jupyter kernel sockets were blocked in the workspace, so a notebook-kernel/UI pass was not claimed. Stored output was not treated as new evidence.

### What is the dataset size and split?

There are 768 rows and eight inputs, with 500 class-0 and 268 class-1 labels. The fixed stratified 80/20 split gives 614 training and 154 test records. The test partition has 100 negatives and 54 positives.

## 2. Preprocessing and leakage

### Why standardise before fitting the SVM?

Feature scales affect the geometry of the separating boundary. The notebook places features on a training-derived standardised scale. This does not guarantee a better model or repair missing measurements, but it avoids letting raw numerical units dominate the geometry.

### Where does the scaler fit?

On training rows only. The audit verified the sample count, training means, disjoint split and predictions. Test data is transformed with those fitted statistics. No scaling leakage was found in this notebook.

> Analogy: a student should prepare from training material and then sit a separate exam. Fitting preprocessing on all rows lets information from the exam influence preparation, even if labels were not explicitly used.

### Does zero explicit missingness mean the data is complete?

No. The audit found zero-coded measurements in five fields, affecting 376 rows. The benchmark's maintained documentation treats those designated zeros as missing-data issues. The notebook does not handle them. Zero pregnancy counts should not be included in the same blanket replacement rule.

### What would a correct imputation design look like?

As a proposed repair, mark designated values missing, split the data, fit an imputer on training rows and apply it to held-out rows. Include it with scaling and the classifier in a Pipeline. The current notebook has no such imputer, and improved metrics must be demonstrated rather than promised.

### Why use a Pipeline during cross-validation?

Each fold needs preprocessing fitted only on its own training partition. A Pipeline makes that sequence easier to preserve. The auditor used one for five-fold evaluation; the original notebook itself uses separate scaler/classifier objects and does not implement CV.

## 3. Model concepts

### What does a linear SVC learn?

A separating hyperplane in feature space, with regularisation and tolerance for errors. Training observations called support vectors help determine the boundary. This implementation selects a linear kernel, not RBF or a deep neural network.

### What does C control?

It changes the trade-off between regularisation and training violations. A larger value penalises violations more strongly; a smaller value favours stronger regularisation. The notebook does not tune C, so do not describe a completed hyperparameter search.

### Is the decision function a probability?

No. It is a classifier score useful for ranking and threshold-based evaluation. A score, a hard class and a calibrated probability are different outputs. The notebook does not establish calibrated individual disease risk.

### Can feature coefficients establish causes?

No. A fitted association in a standardised benchmark model does not establish a causal medical relationship. Interpretation also depends on preprocessing, correlated features, data quality and the population represented.

## 4. Defend the results

### What were the reproduced accuracy values?

Training accuracy was 78.66% and held-out accuracy was 77.27%. Always pair that with the environment, split and other metrics. The majority-class test baseline was 64.94%.

### What was the confusion matrix?

There were 91 true negatives, 9 false positives, 26 false negatives and 28 true positives, using class 1 as positive. Thus 119 of 154 predictions were correct, but only 28 of 54 positive records were detected.

### Calculate precision, recall and specificity.

Positive precision is 28 / (28 + 9) = 75.68%. Positive recall is 28 / (28 + 26) = 51.85%. Specificity is 91 / (91 + 9) = 91%. These answer different questions and should not be interchanged.

### What does balanced accuracy add?

It averages positive recall and specificity, giving 71.43% here. It prevents the more common negative class from dominating the summary in the same way ordinary accuracy can.

### Why is 77.27% not enough for an all-clear?

The model misses 26 positive test records, its setup is not reproducible as published, zero-coded measurements remain untreated and its input sequence accepts invalid values. A benchmark score is not a validated diagnostic system or proof of external generalisation.

### What did five-fold validation show?

The auditor's stratified folds gave mean accuracy 77.73%, with a 2.88 percentage-point fold standard deviation. Mean positive recall was 57.05%. This used the same baseline, not a tuned or imputed replacement. Fold standard deviation is not a 95% confidence interval.

### What did the ten split replays establish?

Changing split seeds from 0 through 9 produced accuracy from 68.83% to 79.87%. This shows sensitivity to sampling; it is not a licence to select the best split as the advertised result. A fixed seed provides reproducibility, not proof of robustness.

### Does the accuracy interval prove deployment readiness?

No. The audit's Wilson interval was 70.04% to 83.18% under its assumptions. It addresses test-sample uncertainty, not missing-data bias, calibration, population shift or prospective clinical utility.

## 5. Input handling and engineering

### What invalid inputs were rejected?

The tested sequence raised library errors for wrong feature counts, non-numeric text, NaN and infinity. These are not a designed application-level validation experience; they come from the underlying libraries.

### What invalid inputs still produced output?

All-zero values and negative glucose, BMI, age or pregnancy count received class-0 predictions. Reversing feature order changed the example to class 1 without a schema-level error. Named fields and explicit value checks are missing.

> Analogy: an eight-item form can have every box filled while the answers are in the wrong boxes. Shape validation checks the number of answers; schema validation checks their meaning and placement.

### Why does inference emit a feature-name warning?

The scaler was fitted with named DataFrame columns but receives an unnamed array for the example. The hard-coded original order matches, but the interface does not protect that order. Preserving validated named columns is a proposed repair.

### Why save preprocessing and model together?

Predictions must use the same fitted transform, feature order and classifier. Saving only the SVM can lose essential context. A versioned preprocessing/model bundle is proposed; the audited repository does not contain one.

### What tests would you add first?

File/schema loading, split separation, training-only fit, designated missing-value behavior, named feature order, malformed-input handling and save/load prediction consistency. Add metric reporting and dependency checks. These are a plan, not an existing passing suite.

### What is the other notebook?

A plant-image project requiring private Drive folders. Its execution could not reach training. Its fit call reuses the training generator as validation, which is a separate validation defect. Do not use its plots to support diabetes performance claims.

## 6. Explain the next step without exaggeration

### Would you immediately add XGBoost or deep learning?

First repair reproducibility, validation, missing-measurement handling and evaluation. A later comparison should use consistent splits, leakage-safe preprocessing and declared objectives. More complex model names do not resolve the current engineering defects.

### How would you investigate the false negatives?

Inspect errors by available feature ranges and missingness, then compare models or thresholds using training/validation data. Keep the final holdout separate. Threshold changes can trade recall against precision and specificity; a desired benefit must be measured with an appropriate objective.

### What should you say about clinical use?

The repository demonstrates classification on a small benchmark. It does not contain prospective validation, an approved clinical workflow or established calibrated risk. Report the measured model behavior without presenting outputs as diagnoses.

### What can you safely put on a resume?

A basic stratified, standardised linear-SVM experiment with a reproducible 77.27% holdout score under the stated compatible environment. Do not claim deployed infrastructure, completed imputation, CNN-based diabetes detection, a calibrated risk service or a high-recall system.

## Practice

Sketch split/fit/transform order; calculate all four confusion-matrix metrics; explain the filename and pandas failures; distinguish zeros from missingness; and explain a hard label versus a decision score versus a calibrated probability.

## Sources

- [Reviewed code](https://github.com/Swapnil-byte-798/Diabetes-prediction-System/tree/f2786e3d5783860d6b40dcec534b5f99037b3f17)
- [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [Benchmark notes](https://search.r-project.org/CRAN/refmans/mlbench/html/PimaIndiansDiabetes.html)
