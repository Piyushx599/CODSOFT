# Titanic Survival Prediction App

import streamlit as st
import pandas as pd
import joblib


model = joblib.load("titanic_model_advanced.pkl")
scaler = joblib.load("titanic_scaler.pkl")


st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Prediction App")
st.markdown("Enter passenger details below and check if they might have survived the Titanic tragedy!")

st.divider()


st.subheader("🧍 Passenger Details")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
    sex = st.selectbox("Gender", ["male", "female"])
    age = st.slider("Age", 0, 80, 25)
    fare = st.number_input("Ticket Fare", 0.0, 600.0, 32.0, step=1.0)

with col2:
    sibsp = st.number_input("Siblings/Spouses Aboard", 0, 8, 0)
    parch = st.number_input("Parents/Children Aboard", 0, 6, 0)
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])




family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0


if age <= 12:
    age_group = 0
elif age <= 18:
    age_group = 1
elif age <= 35:
    age_group = 2
elif age <= 50:
    age_group = 3
else:
    age_group = 4


if fare <= 15:
    fare_bin = 1
elif fare <= 30:
    fare_bin = 2
elif fare <= 100:
    fare_bin = 3
else:
    fare_bin = 4

sex_val = 1 if sex == "male" else 0
embarked_val = {"C": 0, "Q": 1, "S": 2}[embarked]


input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex_val],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked_val],
    "FamilySize": [family_size],
    "IsAlone": [is_alone],
    "AgeGroup": [age_group]
})



scaled_input = scaler.transform(input_data.values)

if st.button("🔮 Predict Survival"):
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1] * 100

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"Passenger would have SURVIVED (Survival Rate: {probability:.2f}%)")
    else:
        st.error(f"Passenger would NOT survive (SurvivalRate: {probability:.2f}%)")

    st.divider()


st.markdown("---")
st.caption("Made by Piyush | A simple ML + Streamlit project 😊")
