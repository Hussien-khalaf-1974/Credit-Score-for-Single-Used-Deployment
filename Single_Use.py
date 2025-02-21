import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler

import os
import subprocess

try:
    import joblib
except ModuleNotFoundError:
    subprocess.run(["pip", "install", "joblib"])
    import joblib


# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Credit Scoring - Banking App", page_icon="🏦")

# 🎯 Load Model & Scaler
@st.cache_resource
def load_model():
    return joblib.load(r"Credit_scoring_classification.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load(r"scaler.pkl")

model = load_model()
scaler = load_scaler()

# 🎯 Frequency Encoding Dictionary
freq_encoding = {
    "Payment_of_Min_Amount": {"Yes": 59432, "No": 40568},
    "Credit_Mix": {"Good": 30384, "Standard": 45848, "Bad": 23768}
}

# 🎯 Function to Preprocess Data
def preprocess_data(payment_of_min_amount, credit_mix, interest_rate, changed_credit_limit, num_bank_accounts, num_credit_card, credit_history_age, age):
    categorical_features = np.array([
        freq_encoding["Payment_of_Min_Amount"].get(payment_of_min_amount, 0.5),
        freq_encoding["Credit_Mix"].get(credit_mix, 0.3)
    ])
    
    numerical_features = np.array([[interest_rate, changed_credit_limit, num_bank_accounts, num_credit_card, credit_history_age]])
    scaled_numerical = scaler.transform(numerical_features)

    return np.concatenate((categorical_features, scaled_numerical.flatten()))

# 🎯 Function to Predict Credit Score & Convert to Category
def predict_credit_score(features):
    # Ensure correct feature length
    if len(features) != 7:
        st.error(f"Expected 7 features, but received {len(features)}. Please check inputs.")
        return None
    
    # Predict credit score
    prediction = model.predict([features])[0]

    # Map prediction to category
    credit_score_mapping = {
        28998: "Poor",
        53174: "Standard",
        17828: "Good"
    }

    return credit_score_mapping.get(prediction , "Unknown")

# 🎯 Streamlit UI
st.title("🏦 Credit Scoring Prediction \n Used With User")

st.header("Enter Your Personal Details")

# 🔹 Personal Information
customer_id = st.text_input("Customer ID", placeholder="Enter your Customer ID")
name = st.text_input("Full Name", placeholder="Enter your full name")
age = st.number_input("Age", min_value=18, max_value=100, step=1)

# 🔹 Financial Information
st.header("Enter Your Financial Details")

payment_of_min_amount = st.selectbox("Payment of Minimum Amount", ["Yes", "No"])
credit_mix = st.selectbox("Credit Mix", ["Good", "Standard", "Bad"])
interest_rate = st.number_input("Interest Rate", min_value=0.0, max_value=50.0, step=0.1)
changed_credit_limit = st.number_input("Changed Credit Limit", min_value=-100000.0, max_value=100000.0, step=100.0)
num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, max_value=50, step=1)
num_credit_card = st.number_input("Number of Credit Cards", min_value=0, max_value=50, step=1)
credit_history_age = st.number_input("Credit History Age (Years)", min_value=0, max_value=100, step=1)

# 🔹 Predict Button
if st.button("Predict Credit Score"):
    if customer_id and name:
        st.write(f"🔄 Processing Prediction for {name} (ID: {customer_id})...")

        # Preprocess Data (Fix includes 'age' in numerical data)
        final_features = preprocess_data(
            payment_of_min_amount, credit_mix, interest_rate, changed_credit_limit, num_bank_accounts, num_credit_card, credit_history_age, age
        )

        predicted_category = predict_credit_score(final_features)

        if predicted_category:
            st.success(f"✅ Predicted Credit Score Category: **{predicted_category}**")
    else:
        st.warning("⚠️ Please enter **Customer ID** and **Name** before proceeding.")

