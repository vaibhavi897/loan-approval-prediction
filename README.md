# Loan Approval Prediction System 🏦

A machine learning application that predicts whether a loan applicant will be approved or rejected based on their demographic and financial profile. The project includes a robust predictive model and an interactive web interface built with Streamlit.

## Overview
This project goes beyond a simple linear model by implementing a **Stacking Classifier** (an ensemble technique) to provide more robust predictions. It also features a real-time web UI that validates inputs and warns users if their data falls out-of-distribution (OOD) compared to the training data.

### Tech Stack
- **Python**: Core programming language
- **Scikit-Learn**: Machine learning modeling, hyperparameter tuning (`GridSearchCV`), and preprocessing (`StandardScaler`)
- **Pandas & NumPy**: Data manipulation and analysis
- **Streamlit**: Interactive web frontend
- **Matplotlib & Seaborn**: Data visualization (Exploratory Data Analysis)

## Model Architecture
The core model is an optimized **Stacking Classifier** consisting of:
1. **Random Forest Classifier**: Captures non-linear relationships.
2. **Gradient Boosting Classifier**: Sequentially builds trees to correct errors of previous ones.
3. **Logistic Regression (Meta-Learner)**: Combines the predictions of the base models to make the final decision.

All models are tuned using `GridSearchCV` (5-fold cross-validation) and the input features are scaled using `StandardScaler`.

**Performance**: The optimized model achieves an accuracy of ~85% and a ROC-AUC of ~84%. 
*(Note: On this specific dataset, `Credit_History` strongly dominates the signal. Attempting to achieve >90% accuracy on this historical dataset typically results in severe overfitting).*

## Key Features
- **End-to-End Pipeline**: Includes a clean `train.py` script for preprocessing data, encoding features, scaling, tuning hyperparameters via Grid Search, and exporting both the model and the scaler.
- **Interactive Web App**: A Streamlit interface (`app.py`) allowing users to input their data and get real-time predictions.
- **Out-of-Distribution (OOD) Warnings**: The app checks if user inputs (like income or loan amount) are significantly higher or lower than the 1st/99th percentiles of the training data, warning the user about potential low prediction confidence.
- **Data Validation**: Categorical inputs are safely mapped to the label encodings expected by the model.

## How to Run

1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **(Optional) Re-train the model**:
   ```bash
   python train.py
   ```
4. **Run the web application**:
   ```bash
   streamlit run app/app.py
   ```

## Project Structure
```text
loan-approval-prediction/
├── app/
│   └── app.py              # Streamlit frontend application
├── data/
│   └── loan_data.csv       # Historical dataset (614 records)
├── models/
│   ├── loan_model.pkl      # Serialized Stacking Classifier
│   └── scaler.pkl          # Serialized StandardScaler
├── notebooks/
│   └── EDA.ipynb           # Exploratory Data Analysis
├── train.py                # Model training and export pipeline
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Key Insights
During the Exploratory Data Analysis (EDA) and modeling phases, it became clear that **Credit History** is the primary driver for loan approval in this dataset. Features like `Education` and `Property Area` have a much weaker correlation with the target variable and are mostly utilized by the model as tie-breakers in borderline financial scenarios.
