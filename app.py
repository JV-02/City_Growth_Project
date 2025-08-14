import streamlit as st
import pandas as pd
import joblib
import json

# Load model and selected features
model = joblib.load('model.pkl')
with open('selected_features.json', 'r') as f:
    selected_features = json.load(f)

st.title("Prediction of Green Cover Percentage using Lasso Model")

# User input for features
st.write("Enter input values for prediction:")
user_input = {}
for feature in selected_features:
    user_input[feature] = st.number_input(f"Enter value for {feature}:", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write(f"Predicted Green Cover Percentage: {prediction[0]:.2f}%")