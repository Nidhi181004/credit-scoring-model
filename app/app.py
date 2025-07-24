import streamlit as st
import joblib
import numpy as np

model = joblib.load('models/random_forest.pkl')

st.title("Credit Scoring Prediction App")
st.write("Enter the applicant's financial data to predict creditworthiness.")

duration = st.number_input("Duration (months)", min_value=4, max_value=72)
amount = st.number_input("Credit Amount", min_value=250, max_value=20000)
installment_rate = st.number_input("Installment Rate (%)", min_value=1, max_value=4)

status = st.selectbox("Status", [0, 1, 2, 3])
credit_history = st.selectbox("Credit History", [0, 1, 2, 3, 4])
purpose = st.selectbox("Purpose", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
savings = st.selectbox("Savings", [0, 1, 2, 3, 4])
employment = st.selectbox("Employment", [0, 1, 2, 3, 4])
age = st.slider("Age", 18, 75)

if st.button("Predict Creditworthiness"):
   
    try:
        debt_income_ratio = amount / (duration * installment_rate)
    except ZeroDivisionError:
        debt_income_ratio = 0

    
    features = np.array([[status, duration, credit_history, purpose, amount,
                          savings, employment, installment_rate, 0, 0, 0, 0,
                          age, 0, 0, 0, 0, 0, 0, 0, debt_income_ratio]])

    prediction = model.predict(features)[0]
    result = " Good Credit" if prediction == 1 else "Bad Credit"

    st.subheader("Prediction Result:")
    st.success(result)
