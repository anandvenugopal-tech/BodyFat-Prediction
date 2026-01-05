import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Body Fat Prediction", layout="centered")


@st.cache_resource
def load_artifacts():
    with open("final_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    with open('onehot.pkl', 'rb') as f:
        onehot = pickle.load(f)

    return model, scaler, feature_columns, onehot


model, scaler, feature_columns, onehot = load_artifacts()


st.markdown(
    """
    <h1 style="text-align:center; color:#2E8B57;">
         Body Fat Prediction
    </h1>
    <p style="text-align:center; font-size:18px; color:gray;">
        Enter your measurements to estimate body fat percentage
    </p>
    <br>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    weight = st.number_input("Weight (kg)", min_value=0.0, step=0.1)
    height = st.number_input("Height (cm)", min_value=0.0, step=0.1)
    abdomen = st.number_input("Abdomen (cm)", min_value=0.0, step=0.1)

with col2:
    chest = st.number_input("Chest (cm)", min_value=0.0, step=0.1)
    hip = st.number_input("Hip (cm)", min_value=0.0, step=0.1)
    thigh = st.number_input("Thigh (cm)", min_value=0.0, step=0.1)
    gender = st.selectbox("Gender", options=["Male", "Female"])


user_input_dict = {
    "Age": age,
    "Weight_kg": weight,
    "Height_cm": height,
    "Abdomen_cm": abdomen,
    "Chest_cm": chest,
    "Hip_cm": hip,
    "Thigh_cm": thigh,
    "Gender_Female": 1.0 if gender == "Female" else 0.0,
    "Gender_Male": 1.0 if gender == "Male" else 0.0
}


final_input_data = [user_input_dict[col] for col in feature_columns]
input_array = np.array(final_input_data).reshape(1, -1)

scaled_input = scaler.transform(input_array)

st.write("")
center_btn = st.columns(3)
with center_btn[1]:
    predict_btn = st.button("🔍 Predict", use_container_width=True)

if predict_btn:
    prediction = model.predict(scaled_input)[0]

    st.markdown(
        f"""
        <div style="
            margin-top: 25px;
            background: #2E8B57;
            padding: 20px;
            border-radius: 12px;
            color: white;
            font-size: 22px;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        ">
            Predicted Body Fat: <b>{prediction:.2f}%</b>
        </div>
        """,
        unsafe_allow_html=True,
    )