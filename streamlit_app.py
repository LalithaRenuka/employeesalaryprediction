import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load model
model = joblib.load("income_classifier_model.joblib")

# Page setup
st.set_page_config(page_title="Income Classifier", layout="centered")

# Custom theme and layout
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Quicksand', sans-serif;
        background: linear-gradient(to bottom right, #e6e6fa, #f8f0ff);
    }

    .stApp {
        padding: 2rem;
    }

    h1 {
        font-weight: 600;
        color: #4B0082;
        margin-bottom: 0.5rem;
    }

    h2, h3, h4 {
        color: #6A0DAD;
    }

    .stForm {
        background-color: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0px 4px 16px rgba(0, 0, 0, 0.1);
    }

    .stSelectbox > div, .stRadio, .stSlider, .stTextInput {
        border-radius: 12px !important;
    }

    .stButton button {
        background: linear-gradient(to right, #8A2BE2, #BA55D3);
        border: none;
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        margin-top: 1rem;
        transition: 0.3s ease;
    }

    .stButton button:hover {
        background: linear-gradient(to right, #7B68EE, #DA70D6);
        transform: scale(1.03);
    }

    .result-box {
        background-color: #fff0f5;
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }

    .result-box h2 {
        color: #800080;
        font-weight: 600;
    }

    </style>
""", unsafe_allow_html=True)

# Title
st.title("Employee Salary Prediction📊")
st.markdown("Predicts whether income is >50K or <=50K based on provided details.")

# Input Form
with st.form("income_form"):
    st.subheader("👤 Enter Personal Details")
    age = st.slider("Age", 17, 90, value=30)
    workclass = st.selectbox("Workclass", [
        'Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov',
        'Federal-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked'
    ])
    education = st.selectbox("Education", [
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
        'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate'
    ])
    marital_status = st.selectbox("Marital Status", [
        'Married-civ-spouse', 'Divorced', 'Never-married',
        'Separated', 'Widowed', 'Married-spouse-absent'
    ])
    occupation = st.selectbox("Occupation", [
        'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
        'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners'
    ])
    gender = st.radio("Gender", ['Male', 'Female'], horizontal=True)
    hours_per_week = st.slider("Hours per Week", 1, 100, value=40)
    native_country = st.selectbox("Native Country", [
        'United-States', 'India', 'Philippines', 'Mexico',
        'Germany', 'Canada', 'England', 'China', 'Other'
    ])
    submitted = st.form_submit_button("💡 Predict Income")

# Prediction & Output
if submitted:
    input_data = pd.DataFrame([{
        'age': age,
        'workclass': workclass,
        'education': education,
        'marital-status': marital_status,
        'occupation': occupation,
        'gender': gender,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }])

    prediction = model.predict(input_data)[0]

    st.markdown(f"""
        <div class="result-box">
            <h2>🔍 Predicted Income Category: {prediction}</h2>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("📊 Model Visualizations")
    st.image(Image.open("confusion_matrix.png"), caption="Confusion Matrix (Red Theme)")
    st.image(Image.open("feature_importance.png"), caption="Top 10 Important Features (Red Theme)")
