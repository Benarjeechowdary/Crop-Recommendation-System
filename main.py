import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🌱 Crop Recommendation System")

N = st.number_input("Nitrogen", min_value=0.0)
P = st.number_input("Phosphorous", min_value=0.0)
K = st.number_input("Potassium", min_value=0.0)
temp = st.number_input("Temperature (°C)", min_value=0.0)
hum = st.number_input("Humidity (%)", min_value=0.0)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0)
rain = st.number_input("Rainfall (mm)", min_value=0.0)

if st.button("Predict"):
    input_data = np.array([[N, P, K, temp, hum, ph, rain]])
    input_scaled = scaler.transform(input_data)   # 🔥 IMPORTANT
    prediction = model.predict(input_scaled)

    st.success(f"🌾 Recommended Crop: {prediction[0]}")
