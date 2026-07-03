import streamlit as st
import pickle
import pandas as pd


import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "models", "loan_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))
# -----------------------------
# App Title
# -----------------------------

st.title("🏦 Loan Approval Prediction System")

st.write("Enter applicant details to check loan eligibility")

# -----------------------------
# USER INPUTS
# -----------------------------

gender = st.selectbox("Gender", ["Male", "Female"])
gender = 1 if gender == "Male" else 0

married = st.selectbox("Married", ["Yes", "No"])
married = 1 if married == "Yes" else 0

education = st.selectbox("Education", ["Graduate", "Not Graduate"])
education = 0 if education == "Graduate" else 1

dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
dependents = 3 if dependents == "3+" else int(dependents)

self_employed = st.selectbox("Self Employed", ["No", "Yes"])
self_employed = 1 if self_employed == "Yes" else 0

property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if property_area == "Urban":
    property_area = 2
elif property_area == "Semiurban":
    property_area = 1
else:
    property_area = 0

# Income inputs (realistic ranges)
applicant_income = st.number_input("Applicant Income", 0, 500000, 5000)
coapplicant_income = st.number_input("Coapplicant Income", 0, 300000, 0)
loan_amount = st.number_input("Loan Amount (in thousands)", 1, 10000, 150)
loan_term = st.number_input("Loan Term (Months)", 12, 480, 360)

# Check for out-of-distribution inputs
ood_warnings = []
if applicant_income > 35000:
    ood_warnings.append("Applicant Income is much higher than most training data.")
if coapplicant_income > 10000:
    ood_warnings.append("Coapplicant Income is much higher than most training data.")
if loan_amount > 500 or loan_amount < 30:
    ood_warnings.append("Loan Amount is outside the typical training data range.")

if ood_warnings:
    st.warning("⚠️ " + " ".join(ood_warnings) + " Prediction confidence may be lower than usual.")



credit_history = st.selectbox("Credit History", [0, 1])

# Feature Engineering
total_income = applicant_income + coapplicant_income

# -----------------------------
# PREDICTION
# -----------------------------

if st.button("Predict Loan Approval"):

    input_data = pd.DataFrame({
        "Gender":[gender],
        "Married":[married],
        "Dependents":[dependents],
        "Education":[education],
        "Self_Employed":[self_employed],
        "ApplicantIncome":[applicant_income],
        "CoapplicantIncome":[coapplicant_income],
        "LoanAmount":[loan_amount],
        "Loan_Amount_Term":[loan_term],
        "Credit_History":[credit_history],
        "Property_Area":[property_area],
        "TotalIncome":[total_income]
    })

    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)[0]
    probability = model.predict_proba(input_data_scaled)[0][1]

    prob_percent = probability * 100

    # Loan decision
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"Approval Probability: {prob_percent:.2f}%")

    # -----------------------------
    # Risk Meter
    # -----------------------------

    if prob_percent > 70:
        st.success("🟢 Risk Level: Low Risk")

    elif prob_percent > 40:
        st.warning("🟡 Risk Level: Medium Risk")

    else:
        st.error("🔴 Risk Level: High Risk")
