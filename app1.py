import streamlit as st
import pandas as pd
import joblib

model = joblib.load("HeartDiseaseKNN.pkl")

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction")
st.write("Enter patient information below.")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain_type = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"]
)
resting_bp = st.number_input(
    "Resting Blood Pressure",
    min_value=0, max_value=250, value=130
)
fasting_bs = st.selectbox("Fasting Blood Sugar", [0, 1])
max_hr = st.number_input(
    "Maximum Heart Rate",
    min_value=50, max_value=250, value=140
)
exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"])
oldpeak = st.number_input(
    "Oldpeak",
    min_value=-5.0, max_value=10.0, value=0.0, step=0.1
)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict", type="primary"):
    patient = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_pain_type],
        "RestingBP": [resting_bp],
        "FastingBS": [fasting_bs],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope]
    })

    prediction = model.predict(patient)[0]
    probabilities = model.predict_proba(patient)[0]

    st.divider()

    if prediction == 1:
        st.error("⚠️ Higher likelihood of heart disease")
    else:
        st.success("✅ Lower likelihood of heart disease")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("No Disease", f"{probabilities[0] * 100:.2f}%")

    with col2:
        st.metric("Heart Disease", f"{probabilities[1] * 100:.2f}%")

    st.caption(
        "This is a machine-learning prediction and is not a medical diagnosis."
    )
