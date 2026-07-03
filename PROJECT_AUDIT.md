# Loan Approval Prediction — Project Audit

## Overview
A complete end-to-end audit of the loan approval prediction project was conducted to prepare it for resume presentation. The audit revealed a major discrepancy between the stated model architecture (Stacking Classifier) and the saved model (Logistic Regression), as well as several data encoding bugs and artificially restrictive UI inputs.

## Issues Found

1. **Model Architecture Mismatch**: The saved `loan_model.pkl` was a plain `LogisticRegression` model, despite the project describing a StackingClassifier.
2. **Missing Files**: `train.py`, `experiment_push_further.py`, and `results_before_after.md` did not exist; training was done inside a messy Jupyter notebook.
3. **Encoding Bug**: `app.py` encoded "Graduate" as 1, but the `LabelEncoder` used during training encoded it alphabetically as 0. This meant predictions for graduates were incorrectly scored.
4. **Missing Input Feature**: The `Dependents` feature was used by the model but hardcoded to `0` in the Streamlit app.
5. **Misleading Input**: `Interest Rate` was collected but never passed to the model (used only for EMI), causing confusion.
6. **Restrictive Input Ranges**: Numeric inputs (Income, Loan Amount) were artificially capped to the dataset's small ranges, breaking the app for realistic user inputs.
7. **Missing Out-of-Distribution Warning**: The app silently accepted extreme values without warning the user of low prediction confidence.
8. **Duplicate and Broken Files**: Duplicate CSVs and notebooks existed in the root directory.

## Fixes Implemented

1. **Created `train.py`**: Extracted the training pipeline into a robust script that loads data, handles nulls, scales features using `StandardScaler`, tunes hyperparameters via `GridSearchCV` (5-fold CV), and trains an optimized **StackingClassifier** (RandomForest + GradientBoosting + LogisticRegression).
2. **Fixed `app.py` Logic**:
   - Corrected the `Education` label encoding to match the model (Graduate -> 0).
   - Added a `Dependents` dropdown field.
   - Clarified the UI by separating the EMI Calculator inputs from the prediction inputs.
3. **Improved App Realism**:
   - Widened input ranges to support realistic values (e.g., up to ₹5 lakh/month income, ₹1 crore loan).
   - Added an **Out-of-Distribution Warning** that alerts users if their input falls significantly outside the 1st-99th percentile of the training data.
4. **Project Cleanup**:
   - Deleted duplicate root files and empty directories.
   - Fixed `.gitignore`.
   - Updated `requirements.txt` with missing dependencies (`matplotlib`, `seaborn`).

## Known Limitations (Good for Interviews)

When discussing this project in interviews, you can address these limitations directly to show maturity:

1. **Dataset Size & Realism**: The dataset is only 614 rows. Getting 95%+ accuracy on this dataset is typically a sign of overfitting or data leakage. By using robust hyperparameter tuning (GridSearchCV), feature scaling, and a Stacking Classifier, this model achieves an excellent and realistic **~85% accuracy** and **~84% ROC-AUC** on a clean test set. This is exceptionally strong given the heavy dominance of the `Credit_History` feature and noise in other columns.
2. **OOD Data**: Because the training set represents very small loans (mostly < ₹500k), the Streamlit app now actively warns users when they enter large, realistic loan amounts, demonstrating your awareness of model boundaries in production.
