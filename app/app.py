import streamlit as st
import pickle
import pandas as pd
import os

# -----------------------------
# Load trained model
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "models", "loan_model.pkl")

model = pickle.load(open(model_path, "rb"))

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
education = 1 if education == "Graduate" else 0

self_employed = st.selectbox("Self Employed", ["No", "Yes"])
self_employed = 1 if self_employed == "Yes" else 0

property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if property_area == "Urban":
    property_area = 2
elif property_area == "Semiurban":
    property_area = 1
else:
    property_area = 0

# Income inputs (restricted to realistic dataset ranges)
applicant_income = st.number_input("Applicant Income", 1000, 25000, 5000)

coapplicant_income = st.number_input("Coapplicant Income", 0, 10000, 0)

loan_amount = st.number_input("Loan Amount (in thousands)", 50, 500, 150)

loan_term = st.number_input("Loan Term (Months)", 12, 480, 360)

interest_rate = st.number_input("Interest Rate (%)", 1.0, 20.0, 8.5)

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
        "Dependents":[0],
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

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

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

    # -----------------------------
    # EMI Calculator
    # -----------------------------

    P = loan_amount * 1000
    R = interest_rate / (12 * 100)
    N = loan_term

    emi = (P * R * (1 + R)**N) / ((1 + R)**N - 1)

    total_payment = emi * N
    total_interest = total_payment - P

    st.subheader("💰 Loan EMI Details")

    st.write(f"Monthly EMI: ₹{emi:.2f}")
    st.write(f"Total Payment: ₹{total_payment:.2f}")
    st.write(f"Total Interest: ₹{total_interest:.2f}")