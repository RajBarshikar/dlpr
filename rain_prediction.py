import pandas as pd
import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model

# ==========================================
# 1. LOAD SAVED ARTIFACTS
# ==========================================
print("Loading model and scaler...")
model = load_model('rain_forecast_lstm.h5')
scaler = joblib.load('scaler.pkl')

# Load the column names used during training
# This is critical to ensure One-Hot Encoding matches exactly
with open('model_columns.json', 'r') as f:
    training_columns = json.load(f)

# ==========================================
# 2. PREPARE NEW DATA
# ==========================================
# In a real scenario, you would load this from a database or API.
# Here, we load the original CSV and take the LAST 14 days as our "recent data"
df_new = pd.read_csv('rain_forecasting assign4.csv')

# Preprocessing (Same steps as training)
df_new['Date'] = pd.to_datetime(df_new['Date'])
df_new = df_new.sort_values(by='Date')

# Select the last 14 days (The sequence needed for prediction)
input_data = df_new.tail(14).copy() 

print(f"Data selected from {input_data['Date'].min()} to {input_data['Date'].max()}")

# ==========================================
# 3. PREPROCESS THE INPUT
# ==========================================
def preprocess_input(data, training_cols, scaler):
    # A. Encode Binary 'RainToday' (Yes/No -> 1/0)
    # We use a manual map to be safe, or load the LabelEncoder if you saved it.
    data['RainToday'] = data['RainToday'].map({'Yes': 1, 'No': 0})
    
    # B. One-Hot Encode Location
    data = pd.get_dummies(data, columns=['Location'])
    
    # C. ALIGN COLUMNS (Crucial Step)
    # 1. Add missing columns with 0 (e.g., if input is only 'Mumbai', add 'Location_Delhi'=0)
    for col in training_cols:
        if col not in data.columns:
            data[col] = 0
            
    # 2. Reorder columns to match training exactly
    # 3. Drop any extra columns that weren't in training
    data = data[training_cols]
    
    # D. Scale Features
    data_scaled = scaler.transform(data)
    
    return data_scaled

# Drop non-feature columns (Date, RainTomorrow) just like in training
# Note: 'RainTomorrow' is what we are predicting, so we don't use it as input
inference_features = input_data.drop(['Date', 'RainTomorrow'], axis=1, errors='ignore')

# Apply preprocessing
processed_sequence = preprocess_input(inference_features, training_columns, scaler)

# ==========================================
# 4. RESHAPE AND PREDICT
# ==========================================
# LSTM expects shape: (Samples, Time Steps, Features)
# We have 1 sample, 14 time steps, and N features
input_reshaped = processed_sequence.reshape(1, 14, processed_sequence.shape[1])

# Run Inference
prediction_prob = model.predict(input_reshaped)[0][0]

# Interpret result
threshold = 0.5
prediction_class = "Yes" if prediction_prob > threshold else "No"

print("\n----------------PREDICTION----------------")
print(f"Probability of Rain Tomorrow: {prediction_prob:.4f}")
print(f"Will it rain? : {prediction_class}")
print("------------------------------------------")