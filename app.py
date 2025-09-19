# app.py
import numpy as np
import streamlit as st
import pandas as pd
import joblib
import os

# ------------------ Load Model & Data ------------------ #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "movie_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "IMDb_Movies_India.csv")

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"❌ Model file not found at {path}. Please run train_model.py first.")
        st.stop()
    return joblib.load(path)

@st.cache_data
def load_data(path):
    df = pd.read_csv(path, encoding="latin1")
    
    # Clean numeric columns (same as train_model.py)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Duration"] = df["Duration"].astype(str).str.replace(" min", "", regex=False)
    df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")
    df["Votes"] = df["Votes"].astype(str).str.replace(",", "", regex=False)
    df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")
    
    return df

model = load_model(MODEL_PATH)
df = load_data(DATA_PATH)

# ------------------ Title ------------------ #
st.set_page_config(page_title="Movie Rating Prediction", page_icon="🎬")
st.title("🎬 Movie Rating Prediction App")
st.write("Predict the IMDb rating of a movie based on its details.")

# ------------------ Sidebar Inputs ------------------ #
st.sidebar.header("Movie Details")

# Top 50 options for dropdowns
def get_top_50(col):
    return df[col].value_counts().nlargest(50).index.tolist() + ["Other"]

director_options = get_top_50("Director")
actor1_options = get_top_50("Actor 1")
actor2_options = get_top_50("Actor 2")
actor3_options = get_top_50("Actor 3")
genre_options = df["Genre"].dropna().unique().tolist() + ["Other"]

# ------------------ User Input Function ------------------ #
# ------------------ User Input Function ------------------ #
def user_input_features():
    # Movie Name
    MovieName = st.sidebar.text_input("Movie Name", "My Movie", key="movie_name")
    
    # Genre
    Genre = st.sidebar.selectbox("Genre", genre_options, key="genre")
    
    # Top 50 Director/Actors dropdowns with numbering
    Director = st.sidebar.selectbox(
        "Director", [f"{i+1}. {name}" for i, name in enumerate(director_options)], key="director"
    )
    Actor1 = st.sidebar.selectbox(
        "Actor 1", [f"{i+1}. {name}" for i, name in enumerate(actor1_options)], key="actor1"
    )
    Actor2 = st.sidebar.selectbox(
        "Actor 2", [f"{i+1}. {name}" for i, name in enumerate(actor2_options)], key="actor2"
    )
    Actor3 = st.sidebar.selectbox(
        "Actor 3", [f"{i+1}. {name}" for i, name in enumerate(actor3_options)], key="actor3"
    )
    
    # Duration (safe defaults)
    if df["Duration"].dropna().empty:
        min_duration, max_duration = 60, 240  # fallback default range
    else:
        min_duration = int(df["Duration"].dropna().min())
        max_duration = int(df["Duration"].dropna().max())
    Duration = st.sidebar.slider("Duration (minutes)", min_duration, max_duration, 120, key="duration")
    
    # Votes
    Votes = st.sidebar.number_input("Votes (IMDb)", 0, 1000000, 5000, key="votes")
    
    # Year of Release (safe defaults)
    if df["Year"].dropna().empty:
        min_year, max_year = 1950, 2025  # fallback default range
    else:
        min_year = int(df["Year"].dropna().min())
        max_year = int(df["Year"].dropna().max())
    Year = st.sidebar.slider("Year of Release", min_year, max_year, 2020, key="year")
    
    # Feature engineering
    Is_Long_Movie = 1 if Duration > 120 else 0
    Votes_log = np.log1p(Votes)
    
    # Remove numbering for model input
    Director = Director.split(". ", 1)[1]
    Actor1 = Actor1.split(". ", 1)[1]
    Actor2 = Actor2.split(". ", 1)[1]
    Actor3 = Actor3.split(". ", 1)[1]
    
    data = {
        "Genre": Genre,
        "Director": Director,
        "Actor 1": Actor1,
        "Actor 2": Actor2,
        "Actor 3": Actor3,
        "Duration": Duration,
        "Votes": Votes_log,
        "Is_Long_Movie": Is_Long_Movie,
        "Year": Year
    }
    return MovieName, pd.DataFrame(data, index=[0])
MovieName, input_df = user_input_features()

# ------------------ Prediction ------------------ #
prediction = model.predict(input_df)[0]

st.image(
    "https://upload.wikimedia.org/wikipedia/commons/6/69/Clapboard_and_cinema_reel.jpg",
    caption="Movie Prediction 🎬"
)

st.subheader("Prediction")
st.write(f"**Movie:** {MovieName}")
st.write(f"**Predicted IMDb Rating:** {prediction:.2f} ⭐")

# ------------------ Footer ------------------ #
st.markdown("---")
st.write("👨‍💻 Developed by Saurav Thakur")
