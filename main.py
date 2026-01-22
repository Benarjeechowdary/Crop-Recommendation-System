import streamlit as st
import joblib
import numpy as np

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="centered"
)

# -------------------- LOAD MODEL & SCALER --------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("crop-model.pkl")
    scaler = joblib.load("crop-scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# -------------------- TITLE --------------------
st.markdown(
    "<h1 style='text-align: center;'>🌱 Crop Recommendation System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Enter soil and climate details to get the best crop recommendation</p>",
    unsafe_allow_html=True
)

st.divider()

# -------------------- INPUT LAYOUT --------------------
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0, step=1.0)
    P = st.number_input("Phosphorous (P)", min_value=0.0, step=1.0)
    K = st.number_input("Potassium (K)", min_value=0.0, step=1.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0)

with col2:
    temp = st.number_input("Temperature (°C)", min_value=0.0)
    hum = st.number_input("Humidity (%)", min_value=0.0)
    rain = st.number_input("Rainfall (mm)", min_value=0.0)

st.divider()

# -------------------- PREDICTION --------------------
if st.button("🌾 Predict Best Crop", use_container_width=True):

    input_data = np.array([[N, P, K, temp, hum, ph, rain]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    classes = model.classes_

    # Top 3 predictions
    top3 = sorted(
        zip(classes, proba),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    st.success(f"✅ **Recommended Crop:** 🌱 **{prediction.upper()}**")

    st.markdown("### 🔮 Prediction Confidence")
    for crop, p in top3:
        st.progress(int(p * 100))
        st.write(f"**{crop}** : {p * 100:.2f}%")

st.divider()

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("ℹ️ About")

    st.write(
        """
        This AI-based system recommends the most suitable crop
        using soil nutrients and climate conditions.

        **Model Used:**
        - Random Forest Classifier

        **Features:**
        - Nitrogen (N)
        - Phosphorous (P)
        - Potassium (K)
        - Temperature
        - Humidity
        - pH
        - Rainfall
        """
    )

    st.markdown("---")
    st.caption("Developed as an ML Project 🚀")
