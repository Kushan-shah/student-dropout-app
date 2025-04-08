import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# ---------------------------
# Set Page Config & Custom CSS for a polished look
# ---------------------------
st.set_page_config(page_title="Student Dropout Prediction", layout="wide")
st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Load Saved Artifacts for Prediction (using caching for performance)
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

numerical_columns = [
    "Previous_qualification_grade", "Admission_grade",
    "Curricular_units_1st_sem_credited", "Curricular_units_1st_sem_enrolled",
    "Curricular_units_1st_sem_evaluations", "Curricular_units_1st_sem_approved",
    "Curricular_units_1st_sem_grade", "Curricular_units_1st_sem_without_evaluations",
    "Curricular_units_2nd_sem_credited", "Curricular_units_2nd_sem_enrolled",
    "Curricular_units_2nd_sem_evaluations", "Curricular_units_2nd_sem_approved",
    "Curricular_units_2nd_sem_grade", "Curricular_units_2nd_sem_without_evaluations",
    "Age_at_enrollment", "Unemployment_rate", "Inflation_rate", "GDP",
    "Grade_Progression", "Attendance_Consistency"
]

categorical_columns = [
    "Marital_status", "Application_mode", "Course", "Previous_qualification",
    "Nacionality", "Mothers_qualification", "Fathers_qualification",
    "Mothers_occupation", "Fathers_occupation", "Age_Group"
]

# ---------------------------
# Sidebar UI Inputs
# ---------------------------
def extract_numeric(s):
    s = s.strip()
    if " – " in s:
        return int(s.split(" – ")[0])
    elif " - " in s:
        return int(s.split(" - ")[0])
    else:
        return int(s)

# Add your Streamlit sidebar inputs here as before...
# [TRUNCATED FOR BREVITY: Include all the sidebar inputs exactly as before.]

# ---------------------------
# Construct Input DataFrame
# ---------------------------
data = {
    # [TRUNCATED: All the user input collection and parsing]
}
input_df = pd.DataFrame(data, index=[0])
input_df["Grade_Progression"] = input_df["Curricular_units_2nd_sem_grade"] - input_df["Curricular_units_1st_sem_grade"]
input_df["Attendance_Consistency"] = input_df["Curricular_units_1st_sem_approved"] / (input_df["Curricular_units_1st_sem_enrolled"] + 1e-5)
input_df["Age_Group"] = pd.cut(input_df["Age_at_enrollment"], bins=[15,20,25,30,35,100], labels=["15-20", "21-25", "26-30", "31-35", "36+"])

# ---------------------------
# Main Section: Prediction & Processed Input Review
# ---------------------------
st.markdown("## Prediction Section")
if st.button("Predict Dropout"):
    input_df["Age_Group"] = input_df["Age_Group"].astype(str)
    for col in categorical_columns:
        input_df[col] = input_df[col].astype(str)

    try:
        input_encoded = encoder.transform(input_df[categorical_columns])
        input_encoded_df = pd.DataFrame(
            input_encoded,
            columns=encoder.get_feature_names_out(categorical_columns),
            index=input_df.index
        )

        input_proc = input_df.drop(columns=categorical_columns)
        input_proc = pd.concat([input_proc.reset_index(drop=True), input_encoded_df.reset_index(drop=True)], axis=1)
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

    except Exception as e:
        st.error(f"Prediction failed due to: {e}")

# ---------------------------
# Optional: Evaluation Tabs
# ---------------------------
# [TRUNCATED: Include your evaluation tabs section unchanged here if needed]
