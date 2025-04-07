import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import base64

# Page configuration
st.set_page_config(page_title="Prediction of Disease", layout="wide", page_icon="🧑‍⚕️")

# ========== Background Image Function ==========
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Add background image (place 'background.jpg' in the same folder)
add_bg_from_local("background.jpg.png")

# ========== Load Models ==========
working_dir = os.path.dirname(os.path.abspath(__file__))
diabetes_model = pickle.load(open(os.path.join(working_dir, 'saved_models', 'diabetes_model.sav'), 'rb'))
heart_disease_model = pickle.load(open(os.path.join(working_dir, 'saved_models', 'heart_data.sav'), 'rb'))
parkinsons_model = pickle.load(open(os.path.join(working_dir, 'saved_models', 'parkinsons_data.sav'), 'rb'))

# Sidebar navigation
with st.sidebar:
    selected = option_menu('Prediction of Disease System',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# ========== Diabetes Prediction ==========
if selected == 'Diabetes Prediction':
    st.title('🩸 Diabetes Prediction')

    with st.form("diabetes_form"):
        col1, col2 = st.columns(2)
        with col1:
            Glucose = st.number_input('Glucose Level', min_value=0.0)
            BMI = st.number_input('BMI', min_value=0.0)
        with col2:
            Age = st.number_input('Age', min_value=1, max_value=110)
            Pregnancies = st.number_input('Number of Pregnancies', min_value=0)

        submitted = st.form_submit_button("Predict")
        if submitted:
            input_data = [Pregnancies, Glucose, 0, 0, 0, BMI, 0, Age]
            prediction = diabetes_model.predict([input_data])[0]
            st.success("✅ The person is diabetic." if prediction == 1 else "❌ The person is not diabetic.")

# ========== Heart Disease Prediction ==========
if selected == 'Heart Disease Prediction':
    st.title('🫀 Heart Disease Prediction')

    with st.form("heart_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input('Age', min_value=1, max_value=100)
            cp = st.number_input('Chest Pain Type (0–3)', min_value=0, max_value=3)
        with col2:
            thalach = st.number_input('Max Heart Rate Achieved', min_value=0)
            oldpeak = st.number_input('Oldpeak (ST depression)', min_value=0.0)

        submitted = st.form_submit_button("Predict")
        if submitted:
            input_data = [age, 1, cp, 120, 200, 0, 1, thalach, 0, oldpeak, 1, 0, 1]
            prediction = heart_disease_model.predict([input_data])[0]
            st.success("✅ The person has heart disease." if prediction == 1 else "❌ No heart disease detected.")

# ========== Parkinson’s Prediction ==========
if selected == 'Parkinsons Prediction':
    st.title("🧠 Parkinson's Disease Prediction")

    with st.form("parkinsons_form"):
        col1, col2 = st.columns(2)
        with col1:
            fo = st.number_input('MDVP:Fo(Hz)', min_value=0.0)
            jitter = st.number_input('MDVP:Jitter(%)', min_value=0.0)
            shimmer = st.number_input('MDVP:Shimmer', min_value=0.0)
        with col2:
            hnr = st.number_input('HNR', min_value=0.0)
            nhr = st.number_input('NHR', min_value=0.0)
            ppe = st.number_input('PPE', min_value=0.0)
        with col1:
            fhi = st.number_input('MDVP:Fhi(Hz)', min_value=0.0)
        with col2:
            flo = st.number_input('MDVP:Flo(Hz)', min_value=0.0)

        submitted = st.form_submit_button("Predict")
        if submitted:
            # Only include the 8 features the model was trained on
            input_data = [fo, fhi, flo, jitter, shimmer, nhr, hnr, ppe]
            prediction = parkinsons_model.predict([input_data])[0]
            st.success("✅ The person has Parkinson's disease." if prediction == 1 else "❌ No Parkinson's disease detected.")

