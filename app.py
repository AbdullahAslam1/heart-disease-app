# Heart Disease Prediction - Streamlit web app
# Uses the SVC trained in the notebook (heart_svc_model.pkl).
# The label encoders are rebuilt from the same fixed category lists used in training,
# so the numbers sent to the model match the notebook exactly.

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

MODEL_FILE = Path(__file__).parent / "heart_svc_model.pkl"

CATEGORY_LISTS = {
    "Gender":         ["Male", "Female"],
    "ChestPain":      ["Typical", "Atypical", "NonAnginal", "Asymptomatic"],
    "FastingSugar":   ["Yes", "No"],
    "ExerciseAngina": ["Yes", "No"],
}
INPUT_COLUMNS = list(CATEGORY_LISTS)

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")


@st.cache_resource
def load_trained_model():
    with open(MODEL_FILE, "rb") as handle:
        return pickle.load(handle)


@st.cache_resource
def make_encoders():
    encoders = {}
    for column, categories in CATEGORY_LISTS.items():
        enc = LabelEncoder()
        enc.fit(categories)
        encoders[column] = enc
    return encoders


def predict_for(answers):
    row = pd.DataFrame([answers])[INPUT_COLUMNS]
    encoders = make_encoders()
    for column in INPUT_COLUMNS:
        row[column] = encoders[column].transform(row[column])
    return row, int(load_trained_model().predict(row)[0])


st.title("❤️ Heart Disease Predictor")
st.write(
    "Answer four questions about a patient and a Support Vector Classifier trained on the "
    "UCI Cleveland heart disease data will predict whether heart disease is present."
)
st.divider()

left, right = st.columns(2)
with left:
    gender = st.selectbox("Gender", CATEGORY_LISTS["Gender"])
    chest_pain = st.selectbox(
        "Chest pain type", CATEGORY_LISTS["ChestPain"],
        help="Typical / Atypical angina, Non-anginal pain, or Asymptomatic (no chest pain)."
    )
with right:
    fasting_sugar = st.selectbox(
        "Fasting blood sugar above 120 mg/dl?", CATEGORY_LISTS["FastingSugar"], index=1
    )
    exercise_angina = st.selectbox(
        "Chest pain during exercise?", CATEGORY_LISTS["ExerciseAngina"], index=1
    )

if st.button("Predict", type="primary", use_container_width=True):
    encoded_row, result = predict_for({
        "Gender": gender,
        "ChestPain": chest_pain,
        "FastingSugar": fasting_sugar,
        "ExerciseAngina": exercise_angina,
    })
    st.divider()
    if result == 1:
        st.error("### Heart disease likely")
    else:
        st.success("### Heart disease unlikely")
    with st.expander("See the encoded numbers sent to the model"):
        st.dataframe(encoded_row, hide_index=True)

st.divider()
st.caption("A learning project for a machine learning course. This is not medical advice.")
