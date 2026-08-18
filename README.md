# Heart Disease Prediction

A machine learning application for predicting the likelihood of heart disease using a K-Nearest Neighbors (KNN) classifier and Streamlit.

## Model

- Algorithm: K-Nearest Neighbors (KNN)
- Hyperparameter tuning: GridSearchCV
- Preprocessing: StandardScaler + OneHotEncoder
- Model: KNN Pipeline
- Deployment: Streamlit

## Features

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Fasting Blood Sugar
- Maximum Heart Rate
- Exercise Angina
- Oldpeak
- ST Slope

## Prediction

The application provides:

- Heart disease prediction
- Probability of no heart disease
- Probability of heart disease

## How to Run

### Navigate to the project

```bash
cd HeartDiseasePredictor

streamlit run app.py


### This application is a machine-learning prediction tool for educational purposes only. It is not a medical diagnosis and should not be used as a substitute for professional medical advice.
