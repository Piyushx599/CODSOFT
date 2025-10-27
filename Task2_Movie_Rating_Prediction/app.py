import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("movie_rating_predictor.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="🎬 Movie Rating Predictor", layout="wide")

st.title("🎬 Movie Rating Prediction App")
st.write("Predict IMDb ratings for Indian movies based on features like genre, duration, director, and votes.")

col1, col2 = st.columns(2)

with col1:
    duration = st.number_input("Movie Duration (minutes)", min_value=60, max_value=240, value=120)
    votes = st.number_input("Number of Votes", min_value=100, max_value=1000000, value=5000)
    year = st.number_input("Release Year", min_value=1950, max_value=2025, value=2020)

with col2:
    genre = st.selectbox("Main Genre", ["Drama", "Comedy", "Action", "Romance", "Thriller", "Unknown"])
    director = st.text_input("Director Name", "Rajkumar Hirani")
    actor1 = st.text_input("Actor 1", "Amitabh Bachchan")
    actor2 = st.text_input("Actor 2", "Shah Rukh Khan")
    actor3 = st.text_input("Actor 3", "Deepika Padukone")

num_actors_known = 3
is_multigenre = 0
log_votes = np.log1p(votes)

input_df = pd.DataFrame({
    "Year": [year],
    "Duration": [duration],
    "Director": [director],
    "Actor 1": [actor1],
    "Actor 2": [actor2],
    "Actor 3": [actor3],
    "Num_Actors_Known": [num_actors_known],
    "Is_MultiGenre": [is_multigenre],
    "Main_Genre": [genre],
    "Log_Votes": [log_votes]
})

encoded_cols = ['Main_Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
input_df[encoded_cols] = encoder.transform(input_df[encoded_cols])

scaled_input = scaler.transform(input_df)

if st.button("Predict Rating"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"⭐ Predicted IMDb Rating: {prediction:.2f}/10")
    st.balloons()
