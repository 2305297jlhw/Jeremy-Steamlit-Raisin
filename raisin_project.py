import streamlit as st
import pandas as pd
import numpy as np
pip install joblib
import joblib

# Load the trained model and transformers
model = joblib.load('grid_search_SVC_PCA_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

# Load dataset to get feature names
raisin = pd.read_csv("Raisin_Dataset.csv")
features = raisin.drop(columns=['Class', 'ConvexArea', 'Area'])
feature_names = features.columns

# Page styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("Raisin Type Prediction")
st.header("Welcome to the Raisin Classifier")
st.markdown("""
    **Instructions:**
    - Use the sliders below to adjust the feature values.
    - The model will predict the type of raisin based on these values.
    - You'll also see the prediction probability.
""")

# Feature sliders
feature_values = {
    name: st.slider(
        name,
        min_value=float(features[name].min()),
        max_value=float(features[name].max()),
        value=float(features[name].mean()),
        step=0.01
    ) for name in feature_names
}

# Prepare input data
input_data = pd.DataFrame([feature_values], columns=feature_names)
input_data_scaled = scaler.transform(input_data)
input_data_pca = pca.transform(input_data_scaled)

# Make prediction
prediction = model.predict(input_data_pca)
prediction_prob = model.predict_proba(input_data_pca)

# Display results
st.write(f"**Predicted class:** {prediction[0]}")
st.write(f"**Prediction probability:** {prediction_prob.max():.2f}")

# Display available classes
st.write("**Available classes in the dataset:**")
st.write(raisin['Class'].unique())
