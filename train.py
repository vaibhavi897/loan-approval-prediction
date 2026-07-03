import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import warnings

def main():
    warnings.filterwarnings('ignore')
    
    # 1. Load Data
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "data", "loan_data.csv")
    df = pd.read_csv(data_path)

    # 2. Preprocess Data
    # Fill missing values with mode/median
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

    # Convert Dependents to integer (3+ becomes 3)
    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

    # Create TotalIncome feature
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

    # Map Target Variable
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

    # Encode categorical features
    le = LabelEncoder()
    cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Drop Loan_ID as it's not a feature
    df.drop('Loan_ID', axis=1, inplace=True)

    # 3. Train-Test Split (with Stratify to maintain class distribution)
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")

    # 4. Feature Scaling (Crucial for Logistic Regression / Gradient Descent)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Hyperparameter Tuning with GridSearchCV
    print("Tuning Random Forest...")
    rf = RandomForestClassifier(random_state=42)
    rf_params = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'min_samples_leaf': [1, 5]}
    rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X_train_scaled, y_train)
    best_rf = rf_grid.best_estimator_

    print("Tuning Logistic Regression...")
    log = LogisticRegression(random_state=42)
    log_params = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2']}
    log_grid = GridSearchCV(log, log_params, cv=5, scoring='accuracy', n_jobs=-1)
    log_grid.fit(X_train_scaled, y_train)
    best_log = log_grid.best_estimator_

    print("Tuning Gradient Boosting...")
    gb = GradientBoostingClassifier(random_state=42)
    gb_params = {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200], 'max_depth': [3, 5]}
    gb_grid = GridSearchCV(gb, gb_params, cv=5, scoring='accuracy', n_jobs=-1)
    gb_grid.fit(X_train_scaled, y_train)
    best_gb = gb_grid.best_estimator_

    # 6. Build Stacking Classifier
    print("Training Final Stacking Classifier...")
    estimators = [
        ('rf', best_rf),
        ('gb', best_gb),
        ('log', best_log)
    ]

    stack_model = StackingClassifier(
        estimators=estimators, 
        final_estimator=LogisticRegression(random_state=42),
        cv=5
    )

    stack_model.fit(X_train_scaled, y_train)

    # 7. Evaluate
    preds = stack_model.predict(X_test_scaled)
    probs = stack_model.predict_proba(X_test_scaled)[:, 1]

    print("\nOptimized Stacking Classifier Performance:")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall:    {recall_score(y_test, preds):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, probs):.4f}")

    # 8. Save the model AND the scaler
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "loan_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(stack_model, f)
        
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"\nModel successfully saved to {model_path}")
    print(f"Scaler successfully saved to {scaler_path}")

if __name__ == '__main__':
    main()
