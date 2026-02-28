import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Insurance Fraud Detector", page_icon="🚗🛡️")
st.title("🚗 Car Insurance Fraud Detection")
st.markdown("---")

# --- MAPPING DICTIONARIES ---

state_map = {'CA': 0, 'IL': 1, 'NY': 2, 'OH': 3, 'PA': 4} # Example values
sex_map = {'FEMALE': 0, 'MALE': 1, 'OTHER': 2}
edu_map = {'High School': 0, 'College': 1, 'Masters': 2, 'PhD': 3}
incident_map = {'Parked Car': 0, 'Vehicle Theft': 1, 'Single Vehicle Collision': 2, 'Multi-vehicle Collision': 3}
severity_map = {'Trivial Damage': 1, 'Minor Damage': 2, 'Major Damage': 3, 'Total Loss': 4}

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal & Policy Info")
    # User dropdown
    state_select = st.selectbox('Policy State', list(state_map.keys()))
    age = st.slider('Insured Age', 18, 100, 35)
    sex_select = st.selectbox('Insured Sex', list(sex_map.keys()))
    edu_select = st.selectbox('Education Level', list(edu_map.keys()))
    premium = st.number_input('Annual Premium ($)', value=1000.0)
    deductible = st.number_input('Deductible ($)', value=500)

with col2:
    st.subheader("Incident & Claim Info")
    incident_select = st.selectbox('Incident Type', list(incident_map.keys()))
    severity_select = st.selectbox('Incident Severity', list(severity_map.keys()))
    num_vehicles = st.slider('Vehicles Involved', 1, 4, 1)
    witnesses = st.slider('Witnesses', 0, 3, 0)
    claim_amt = st.number_input('Claim Amount ($)', value=5000.0)
    total_claim = st.number_input('Total Claim Amount ($)', value=7000.0)

# Prediction Logic
if st.button("Predict Fraud Status"):
    input_data = np.array([[
        state_map[state_select], deductible, premium, age,
        sex_map[sex_select], edu_map[edu_select], 0, 0, # Dummy values for missing logic
        incident_map[incident_select], 0, severity_map[severity_select], 0,
        12, num_vehicles, 0, witnesses,
        1, claim_amt, total_claim
    ]])
    
    # Scaling and Prediction
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    
    if prediction[0] == 1:
        st.error("⚠️ Result: Potential Fraud Detected!")
    else:
        st.success("✅ Result: Fraud Not Dected..Claim is Legitimate.")

# Observation:
# The Streamlit app provides a user-friendly interface for car insurance fraud detection. Users input personal, policy, and incident details, which are then mapped, scaled, and fed into the trained XGBoost model. The app quickly predicts whether a claim is potentially fraudulent (⚠️) or legitimate (✅), making it easy for insurers to flag suspicious claims in real time.
