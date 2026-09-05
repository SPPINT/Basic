import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# Load all 6 saved artifacts ONCE when the app starts
# (not inside the function - loading files on every click would be slow)
# ---------------------------------------------------------
model = joblib.load('model.pkl')
sc = joblib.load('scaler.pkl')
fake_zero_medians = joblib.load('fake_zero_medians.pkl')
outlier_bounds = joblib.load('outlier_bounds.pkl')
skew_cols = joblib.load('skew_cols.pkl')
final_features = joblib.load('final_features.pkl')


def predict_diabetes(raw_input: dict):
    """
    raw_input: a dict with keys matching the original column names, e.g.
    {'Pregnancies': 2, 'Glucose': 120, 'BloodPressure': 70, 'SkinThickness': 20,
     'Insulin': 85, 'BMI': 28.5, 'DiabetesPedigreeFunction': 0.5, 'Age': 33}

    Applies the exact same steps used in training, in the same order:
    fake-zero fix -> skew fix -> outlier capping -> select features -> scale -> predict
    """
    row = pd.DataFrame([raw_input])

    # 1. fake-zero fix (using the medians saved from training)
    for col, med in fake_zero_medians.items():
        if row.loc[0, col] == 0:
            row.loc[0, col] = med

    # 2. skew fix (same columns, same transform, as training)
    for col in skew_cols['strong']:
        row[col] = np.sign(row[col]) * np.log1p(abs(row[col]))
    for col in skew_cols['mild']:
        row[col] = np.sign(row[col]) * np.sqrt(abs(row[col]))

    # 3. outlier capping (using the bounds saved from training)
    for col, b in outlier_bounds.items():
        row[col] = row[col].clip(lower=b['lower'], upper=b['upper'])

    # 4. select final features, in the correct order
    row = row[final_features]

    # 5. scale (using the already-fitted scaler, never re-fit)
    row_scaled = sc.transform(row)

    # 6. predict
    prediction = model.predict(row_scaled)[0]
    probability = model.predict_proba(row_scaled)[0][1]

    return {'prediction': int(prediction), 'probability': round(float(probability), 4)}


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺")

st.title("🩺 Diabetes Risk Predictor")
st.write(
    "Enter patient details below. This uses a Logistic Regression model "
    "trained on the Pima Indians Diabetes dataset."
)

# Two columns just to make the form look neater
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2, step=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=85)
    bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=28.5, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Age", min_value=1, max_value=120, value=33, step=1)

st.caption(
    "Note: a value of 0 for Glucose, Blood Pressure, Skin Thickness, Insulin, or BMI "
    "is treated as missing and will be filled in automatically using training-data medians, "
    "the same way it was handled during model training."
)

if st.button("Predict", type="primary"):
    raw_input = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    result = predict_diabetes(raw_input)

    st.divider()
    if result['prediction'] == 1:
        st.error(f" Higher risk of diabetes  —  probability: {result['probability']:.1%}")
    else:
        st.success(f" Lower risk of diabetes  —  probability: {result['probability']:.1%}")

    st.progress(result['probability'])
    st.caption(
        "This is a prediction from a machine learning model trained on a public dataset "
        "for educational purposes — it is not a medical diagnosis."
    )
