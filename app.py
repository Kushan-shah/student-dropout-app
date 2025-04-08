import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# ---------------------------
# Page Config & Style
# ---------------------------
st.set_page_config(page_title="Student Dropout Prediction", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Load Artifacts
# ---------------------------
@st.cache_resource
def load_artifacts():
    with open("model/random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model/onehot_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("model/training_columns.pkl", "rb") as f:
        training_columns = pickle.load(f)
    return model, scaler, encoder, training_columns

model, scaler, encoder, training_columns = load_artifacts()

# ---------------------------
# Inputs
# ---------------------------
st.markdown("<h1 style='text-align: center;'>🎓 Student Dropout Prediction</h1>", unsafe_allow_html=True)
st.sidebar.header("Input Student Details")

marital_status = st.sidebar.selectbox("Marital Status", [1, 2, 3, 4, 5, 6])
application_mode = st.sidebar.selectbox("Application Mode", [1, 2, 5, 7, 10, 15, 16, 17, 18, 26, 27, 39, 42, 43, 44, 51, 53, 57])
application_order = st.sidebar.number_input("Application Order", min_value=0, max_value=9, value=0, step=1)
course = st.sidebar.selectbox("Course", [33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254, 9500, 9556, 9670, 9773, 9853, 9991])
daytime_evening_attendance = st.sidebar.selectbox("Attendance", [1, 0])
previous_qualification = st.sidebar.selectbox("Previous Qualification Code", [1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 19, 38, 39, 40, 42, 43])
previous_qualification_grade = st.sidebar.slider("Previous Qualification Grade", 0.0, 200.0, 150.0)
admission_grade = st.sidebar.slider("Admission Grade", 0.0, 200.0, 150.0)
nacionality = st.sidebar.selectbox("Nationality", [1, 2, 6, 11, 13, 14, 17, 21, 22, 24, 25, 26, 32, 41, 62, 100, 101, 103, 105, 108, 109])
mothers_qualification = st.sidebar.selectbox("Mother's Qualification", list(range(1, 45)))
fathers_qualification = st.sidebar.selectbox("Father's Qualification", list(range(1, 45)))
mothers_occupation = st.sidebar.selectbox("Mother's Occupation", list(range(0, 145)))
fathers_occupation = st.sidebar.selectbox("Father's Occupation", list(range(0, 145)))

curricular_units_1st_sem_credited = st.sidebar.number_input("1st Sem Credited", 0, 100, 30)
curricular_units_1st_sem_enrolled = st.sidebar.number_input("1st Sem Enrolled", 0, 100, 30)
curricular_units_1st_sem_evaluations = st.sidebar.number_input("1st Sem Evaluations", 0, 100, 30)
curricular_units_1st_sem_approved = st.sidebar.number_input("1st Sem Approved", 0, 100, 30)
curricular_units_1st_sem_grade = st.sidebar.slider("1st Sem Grade", 0.0, 20.0, 10.0)
curricular_units_1st_sem_without_evaluations = st.sidebar.number_input("1st Sem Without Evaluations", 0, 100, 0)

curricular_units_2nd_sem_credited = st.sidebar.number_input("2nd Sem Credited", 0, 100, 30)
curricular_units_2nd_sem_enrolled = st.sidebar.number_input("2nd Sem Enrolled", 0, 100, 30)
curricular_units_2nd_sem_evaluations = st.sidebar.number_input("2nd Sem Evaluations", 0, 100, 30)
curricular_units_2nd_sem_approved = st.sidebar.number_input("2nd Sem Approved", 0, 100, 30)
curricular_units_2nd_sem_grade = st.sidebar.slider("2nd Sem Grade", 0.0, 20.0, 10.0)
curricular_units_2nd_sem_without_evaluations = st.sidebar.number_input("2nd Sem Without Evaluations", 0, 100, 0)

unemployment_rate = st.sidebar.number_input("Unemployment Rate", 0.0, 50.0, 10.0)
inflation_rate = st.sidebar.number_input("Inflation Rate", -10.0, 50.0, 1.0)
GDP = st.sidebar.number_input("GDP", -10.0, 50.0, 1.0)

# Binary Options
displaced = st.sidebar.selectbox("Displaced", ["Yes", "No"])
educational_special_needs = st.sidebar.selectbox("Educational Special Needs", ["Yes", "No"])
debtor = st.sidebar.selectbox("Debtor", ["Yes", "No"])
tuition_fees_up_to_date = st.sidebar.selectbox("Tuition Fees Up to Date", ["Yes", "No"])
scholarship_holder = st.sidebar.selectbox("Scholarship Holder", ["Yes", "No"])
international = st.sidebar.selectbox("International", ["Yes", "No"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age_at_enrollment = st.sidebar.number_input("Age at Enrollment", 17, 100, 18)

# ---------------------------
# Build Feature Input
# ---------------------------
data = {
    "Marital_status": marital_status,
    "Application_mode": application_mode,
    "Application_order": application_order,
    "Course": course,
    "Daytime_evening_attendance": daytime_evening_attendance,
    "Previous_qualification": previous_qualification,
    "Previous_qualification_grade": previous_qualification_grade,
    "Admission_grade": admission_grade,
    "Nacionality": nacionality,
    "Mothers_qualification": mothers_qualification,
    "Fathers_qualification": fathers_qualification,
    "Mothers_occupation": mothers_occupation,
    "Fathers_occupation": fathers_occupation,
    "Curricular_units_1st_sem_credited": curricular_units_1st_sem_credited,
    "Curricular_units_1st_sem_enrolled": curricular_units_1st_sem_enrolled,
    "Curricular_units_1st_sem_evaluations": curricular_units_1st_sem_evaluations,
    "Curricular_units_1st_sem_approved": curricular_units_1st_sem_approved,
    "Curricular_units_1st_sem_grade": curricular_units_1st_sem_grade,
    "Curricular_units_1st_sem_without_evaluations": curricular_units_1st_sem_without_evaluations,
    "Curricular_units_2nd_sem_credited": curricular_units_2nd_sem_credited,
    "Curricular_units_2nd_sem_enrolled": curricular_units_2nd_sem_enrolled,
    "Curricular_units_2nd_sem_evaluations": curricular_units_2nd_sem_evaluations,
    "Curricular_units_2nd_sem_approved": curricular_units_2nd_sem_approved,
    "Curricular_units_2nd_sem_grade": curricular_units_2nd_sem_grade,
    "Curricular_units_2nd_sem_without_evaluations": curricular_units_2nd_sem_without_evaluations,
    "Unemployment_rate": unemployment_rate,
    "Inflation_rate": inflation_rate,
    "GDP": GDP,
    "Displaced": 1 if displaced == "Yes" else 0,
    "Educational_special_needs": 1 if educational_special_needs == "Yes" else 0,
    "Debtor": 1 if debtor == "Yes" else 0,
    "Tuition_fees_up_to_date": 1 if tuition_fees_up_to_date == "Yes" else 0,
    "Scholarship_holder": 1 if scholarship_holder == "Yes" else 0,
    "International": 1 if international == "Yes" else 0,
    "Gender": 1 if gender == "Male" else 0,
    "Age_at_enrollment": age_at_enrollment
}

input_df = pd.DataFrame(data, index=[0])

# Safely add engineered features
try:
    input_df["Grade_Progression"] = input_df["Curricular_units_2nd_sem_grade"] - input_df["Curricular_units_1st_sem_grade"]
    input_df["Attendance_Consistency"] = input_df["Curricular_units_1st_sem_approved"] / (input_df["Curricular_units_1st_sem_enrolled"] + 1e-5)
except Exception as e:
    st.warning(f"Engineered feature issue: {e}")

# ---------------------------
# Prediction Section
# ---------------------------
st.markdown("## Prediction Section")
if st.button("Predict Dropout"):
    # Only pass columns encoder was trained on
    cat_cols = [
        "Marital_status", "Application_mode", "Course",
        "Nacionality", "Mothers_qualification", "Fathers_qualification",
        "Mothers_occupation", "Fathers_occupation"
    ]
    input_encoded = encoder.transform(input_df[cat_cols])
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(cat_cols), index=input_df.index)

    input_proc = input_df.drop(columns=cat_cols, errors="ignore")
    input_proc = pd.concat([input_proc, input_encoded_df], axis=1)

    input_proc[numerical_columns] = scaler.transform(input_proc[numerical_columns])
    input_proc = input_proc[training_columns]

    prediction = model.predict(input_proc)
    status_map = {0: "Dropout", 1: "Graduate", 2: "Enrolled"}

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prediction Result")
        st.success(f"Predicted Academic Status: **{status_map[prediction[0]]}**")
    with col2:
        st.subheader("Processed Input Features")
        st.dataframe(input_proc)
