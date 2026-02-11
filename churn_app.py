import joblib
import streamlit as st
import numpy as np
import pandas as pd

## Load trained model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

## Streamlit app
st.title("Telecom Customer Churn Prediction")

st.write("Predict whether a customer will churn based on their profile")

## Define input options
genders = ['Male', 'Female']
yes_no = ['Yes', 'No']
internet_services = ['DSL', 'Fiber optic', 'No']
contracts = ['Month-to-month', 'One year', 'Two year']
payment_methods = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']

## User inputs - Demographics
st.subheader("Customer Demographics")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", genders)
    senior_citizen = st.selectbox("Senior Citizen", yes_no)

with col2:
    partner = st.selectbox("Has Partner", yes_no)
    dependents = st.selectbox("Has Dependents", yes_no)

## Account Information
st.subheader("Account Information")
col3, col4 = st.columns(2)

with col3:
    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
    contract = st.selectbox("Contract Type", contracts)

with col4:
    monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0)
    paperless_billing = st.selectbox("Paperless Billing", yes_no)

## Services
st.subheader("Services Subscribed")
col5, col6 = st.columns(2)

with col5:
    phone_service = st.selectbox("Phone Service", yes_no)
    multiple_lines = st.selectbox("Multiple Lines", yes_no + ['No phone service'])
    internet_service = st.selectbox("Internet Service", internet_services)
    online_security = st.selectbox("Online Security", yes_no + ['No internet service'])

with col6:
    online_backup = st.selectbox("Online Backup", yes_no + ['No internet service'])
    device_protection = st.selectbox("Device Protection", yes_no + ['No internet service'])
    tech_support = st.selectbox("Tech Support", yes_no + ['No internet service'])
    streaming_tv = st.selectbox("Streaming TV", yes_no + ['No internet service'])

streaming_movies = st.selectbox("Streaming Movies", yes_no + ['No internet service'])
payment_method = st.selectbox("Payment Method", payment_methods)

## Calculate TotalCharges
total_charges = tenure * monthly_charges

## Predict button
if st.button("Predict Churn Risk"):

    ## Create DataFrame with input
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    ## Preprocess input data
    # Binary encoding
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        input_data[col] = input_data[col].map({'Yes': 1, 'No': 0})
    
    # Gender encoding
    input_data['gender'] = input_data['gender'].map({'Male': 1, 'Female': 0})
    
    # One-Hot Encoding for categorical features
    categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    
    input_data = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)
    
    # Align with training features
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)
    
    # Scale features
    input_scaled = scaler.transform(input_data)
    
    ## Make prediction
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    churn_probability = prediction_proba[1] * 100
    
    ## Display results
    st.markdown("---")
    st.subheader("Prediction Result")
    
    if prediction == 1:
        st.error(f"HIGH RISK - Customer likely to CHURN")
        st.write(f"Churn Probability: **{churn_probability:.1f}%**")
        st.write("**Recommendation:** Target for retention campaign")
    else:
        st.success(f"LOW RISK - Customer likely to STAY")
        st.write(f"Churn Probability: **{churn_probability:.1f}%**")
        st.write("**Recommendation:** No immediate action needed")
    
    ## Show probability bar
    st.progress(int(churn_probability))

## Page styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """,
    unsafe_allow_html=True
)