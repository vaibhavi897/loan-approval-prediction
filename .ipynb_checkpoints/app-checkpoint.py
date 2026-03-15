import streamlit as st
import pickle
import pandas as pd
import os

# Load model
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "models", "loan_model.pkl")

model = pickle.load(open(model_path, "rb"))

# App title
st.title("🏦 Loan Approval Prediction")

st.write("Enter applicant details to check loan eligibility")

# User inputs
applicant_income = st.number_input("Applicant Income", 0, 50000, 5000)
coapplicant_income = st.number_input("Coapplicant Income", 0, 50000, 0)
loan_amount = st.number_input("Loan Amount", 0, 1000, 150)
loan_term = st.number_input("Loan Term (Months)", 0, 480, 360)
credit_history = st.selectbox("Credit History", [0, 1])

# Feature engineering
total_income = applicant_income + coapplicant_income

# Prediction button
if st.button("Predict Loan Approval"):

    input_data = pd.DataFrame({
        "Gender":[1],
        "Married":[1],
        "Dependents":[0],
        "Education":[1],
        "Self_Employed":[0],
        "ApplicantIncome":[applicant_income],
        "CoapplicantIncome":[coapplicant_income],
        "LoanAmount":[loan_amount],
        "Loan_Amount_Term":[loan_term],
        "Credit_History":[credit_history],
        "Property_Area":[1],
        "TotalIncome":[total_income]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"Approval Probability: {probability*100:.2f}%")