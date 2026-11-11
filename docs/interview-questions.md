# Diabetes Prediction System interview questions

Updated 11 November 2026. Practise answers using the numbers and commands from the run.

## Starting the conversation

### Interviewer. What did you build?

I built a Python benchmark that fits a linear support vector classifier to 768 Pima diabetes records. The workflow validates the CSV, runs a stratified evaluation and saves one fitted preprocessing and model artifact. I can score a named JSON record from the command line or walk through the same functions in the notebook.

### Interviewer. What failed when you first ran it?

The notebook expected a CSV filename while the committed CSV bytes had an XLS extension. Copying it to the expected name exposed a second failure in pandas 3.1. The notebook passed both the column and axis arguments to drop. The repaired loader uses the correct filename and selects columns by name. I added pinned dependencies and verified the workflow on pandas 2.2.3 and 3.1.0.

### Interviewer. What is a feature and what is a label?

A feature is an input, such as age or BMI. Outcome is the binary label used during training and evaluation. A prediction request sends only the eight features. The input validator rejects an extra Outcome field.

## Preprocessing and leakage

### Interviewer. Why can zero mean different things?

A pregnancy count of zero is valid. The dataset documents zero as a missing-measurement code in five other columns, including glucose and insulin. I convert those codes into missing values and fill them with training-set medians.

### Interviewer. How did you verify training-only preprocessing?

I checked that the 614 training and 154 holdout indices are disjoint. I compared the saved medians to the training medians and checked that the scaler's means match the imputed training matrix. During cross-validation, each fold fits its own preprocessing and classifier.

### Interviewer. What does scaling do for the SVM?

It puts measurements with different units on a comparable numerical scale. The saved training means and standard deviations also transform later records, so the classifier sees features in the same coordinate system.

### Interviewer. Is the margin a probability?

The margin is the signed linear decision score. I use it for ranking and ROC AUC. An individual risk probability would require separate calibration and validation.

## Interpreting the result

### Interviewer. How did it perform?

The model got 119 out of 154 holdout records right, or 77.27%. Predicting class 0 every time would get 64.94%. The model detected 28 out of 54 positives, giving 51.85% recall. It missed 26 positives, which matters even when overall accuracy looks reasonable.

### Interviewer. Work out precision and specificity.

Of the 37 predicted positives, 28 were true positives, so precision was 28 / 37 = 75.68%. The model correctly classified 91 of the 100 actual negatives, so specificity was 91%. Averaging specificity and recall gives 71.43% balanced accuracy.

### Interviewer. Did adding imputation fix the false negatives?

The confusion matrix on the historical split stayed at 91 true negatives, 9 false positives, 26 false negatives and 28 true positives. ROC AUC increased from 0.7919 to 0.8200. AUC measures score ranking across thresholds. The false negatives remained at the default threshold.

### Interviewer. Why keep the original holdout?

It lets me compare the repaired implementation with the earlier result on the same records. Since that split had already been examined, I describe it as a replay. Five-fold validation runs on the training portion. A separate dataset would be needed for external validation.

## Engineering and next steps

### Interviewer. What did the input tests cover?

I tested reordered JSON fields, a missing measurement, missing and extra field names, strings, non-finite numbers, negative values, duplicate keys and a missing model file. The CLI also trains and scores in separate processes from a fresh directory.

### Interviewer. Why save a pipeline?

Saving the imputer, scaler and classifier together preserves the exact transformations used in training. The validation code turns an incoming object into the expected feature frame before the saved pipeline runs.

### Interviewer. How would you investigate the low recall?

I would set the target trade-off on development data, compare precision and recall across thresholds and check performance by missing-measurement pattern. I would then evaluate the selected rule on independent data. The current score is a small benchmark result with substantial missed positives.

## Sources

- [Execution audit](verification.md)
- [Architecture guide](how-it-works.md)
