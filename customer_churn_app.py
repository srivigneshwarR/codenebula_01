# -*- coding: utf-8 -*-
"""
Spyder Editor - Customer Churn Test Script
"""

import pandas as pd
import numpy as np
import pickle

# Load the saved model
with open('C:/Users/Emayavan/OneDrive/Desktop/Project/customer_churn_model.sav', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Sample input
input_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Load encoders
with open("C:/Users/Emayavan/OneDrive/Desktop/Project/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Encode categorical features
for column, encoder in encoders.items():
    input_df[column] = encoder.transform(input_df[column])

# Predict
prediction = loaded_model.predict(input_df)
prediction_proba = loaded_model.predict_proba(input_df)

print(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
print(f"Prediction Probability: {prediction_proba[0]}")
