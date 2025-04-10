import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import random

# Load pre-trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = load_model('mlp_model.keras')  # Your pre-trained model
    scaler = joblib.load('scaler.pkl')     # Load the scaler saved during training
    return model, scaler

# Normalization function using pre-fitted MinMaxScaler
def normalize_user_input(input_data, scaler, expected_columns):
    # Ensure input has all columns, fill missing with 0
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Select only expected columns
    input_data = input_data[expected_columns].copy()
    
    # Normalize using the pre-fitted scaler (like transform in training)
    normalized_input = scaler.transform(input_data)
    
    return pd.DataFrame(normalized_input, columns=expected_columns)

# Prediction function
def predict_condition(model, data):
    pred_proba = model.predict(data, verbose=0)
    pred_class = np.argmax(pred_proba, axis=1)[0]
    labels = {0: "Normal", 1: "Not Good", 2: "Severe"}
    return labels[pred_class]

# Streamlit App
st.title("Transformer Condition Predictor")

# Sidebar for option selection
option = st.sidebar.selectbox("Choose Input Method", ["Upload CSV", "Manual Input"])

# Expected columns
expected_columns = ['Temperature (°C)', 'Oil Level (%)', 'Current (A)', 'Voltage (V)', 'Vibration (m/s²)']

# Load the pre-trained model and scaler
model, scaler = load_model_and_scaler()

# Option 1: Upload CSV
if option == "Upload CSV":
    st.subheader("Upload Your CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Check columns
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            st.warning(f"Missing columns: {missing_cols}. Filling with 0.")
        
        # Normalize the data
        normalized_df = normalize_user_input(df, scaler, expected_columns)
        
        # Pick a random row
        random_idx = random.randint(0, len(normalized_df) - 1)
        random_row = normalized_df.iloc[random_idx:random_idx+1]
        st.write("Random Row Selected for Prediction:")
        st.write(random_row)
        
        # Predict
        condition = predict_condition(model, random_row.values)
        
        # Display result with color
        if condition == "Normal":
            st.markdown(f"<h3 style='color:green'>Condition: {condition}</h3>", unsafe_allow_html=True)
        elif condition == "Not Good":
            st.markdown(f"<h3 style='color:yellow'>Condition: {condition}</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color:red'>Condition: {condition}</h3>", unsafe_allow_html=True)

# Option 2: Manual Input
else:
    st.subheader("Enter Transformer Parameters Manually")
    
    # Input fields
    temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=150.0, value=50.0)
    oil = st.number_input("Oil Level (%)", min_value=0.0, max_value=100.0, value=80.0)
    current = st.number_input("Current (A)", min_value=0.0, max_value=1000.0, value=300.0)
    voltage = st.number_input("Voltage (V)", min_value=0.0, max_value=300.0, value=230.0)
    vibration = st.number_input("Vibration (m/s²)", min_value=0.0, max_value=5.0, value=0.5)
    
    if st.button("Predict"):
        # Create DataFrame from inputs
        input_data = pd.DataFrame([[temp, oil, current, voltage, vibration]], columns=expected_columns)
        
        # Normalize the data
        normalized_input = normalize_user_input(input_data, scaler, expected_columns)
        st.write("Normalized Input Data:")
        st.write(normalized_input)
        
        # Predict
        condition = predict_condition(model, normalized_input.values)
        
        # Display result with color
        if condition == "Normal":
            st.markdown(f"<h3 style='color:green'>Condition: {condition}</h3>", unsafe_allow_html=True)
        elif condition == "Not Good":
            st.markdown(f"<h3 style='color:yellow'>Condition: {condition}</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color:red'>Condition: {condition}</h3>", unsafe_allow_html=True)
