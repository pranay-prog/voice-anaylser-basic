import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import pandas as pd
from collections import Counter

st.set_page_config(page_title="AI Voice Mood Analyzer", page_icon="🎤", layout="wide")

MOODS = ["Happy", "Sad", "Angry", "Stressed", "Neutral", "Excited"]
emoji_dict = {"Happy": "😃", "Sad": "😢", "Angry": "😡", "Stressed": "😓", "Neutral": "😐", "Excited": "🤩"}
color_dict = {"Happy": "#FFF176", "Sad": "#90CAF9", "Angry": "#EF5350", "Stressed": "#FFB74D", "Neutral": "#E0E0E0", "Excited": "#BA68C8"}

def predict_mood(features):
    # placeholder random predictor — replace with real model later
    probs = np.random.dirichlet(np.ones(len(MOODS)), size=1)[0]
    mood_index = np.argmax(probs)
    return MOODS[mood_index], probs

def extract_features(filepath):
    # librosa expects a filename or file-like object that soundfile can read.
    y, sr = librosa.load(filepath, sr=16000, mono=True)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled.reshape(1, -1), y, sr

def set_bg_color(color):
    # Basic background color for the app container
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {color};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<h1 style='text-align: center;'>🎤 AI Voice Mood Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a short audio clip (5–10 sec) and discover your mood instantly.</p>", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state["history"] = []

uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Save uploaded file to a temp file so librosa/soundfile can read it reliably
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Play audio
    st.audio(tmp_path)

    # Extract features & waveform
    try:
        features, y, sr = extract_features(tmp_path)
    except Exception as e:
        st.error(f"Could not read audio file: {e}")
    else:
        # Predict (placeholder)
        mood, probs = predict_mood(features)
        st.session_state["history"].append(mood)
        set_bg_color(color_dict.get(mood, "#FFFFFF"))

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"<h2 style='color:#333;'>🧠 Mood Detected</h2>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:#111;'>{mood} {emoji_dict[mood]}</h3>", unsafe_allow_html=True)

            st.markdown("<h4>📊 Confidence Scores</h4>", unsafe_allow_html=True)
            for i, m in enumerate(MOODS):
                # progress needs a float between 0 and 1
                st.progress(float(probs[i]))
                st.write(f"{m}: {probs[i]*100:.2f}%")

        with col2:
            st.markdown("<h4>🔊 Waveform</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 2))
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set(yticks=[])
            st.pyplot(fig, clear_figure=True)

            st.markdown("<h4>🌈 Spectrogram</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 2))
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", ax=ax, cmap="magma")
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            st.pyplot(fig, clear_figure=True)

        # Show mood history as a bar chart (counts)
        st.markdown("<h3>📈 Mood History</h3>", unsafe_allow_html=True)
        if st.session_state["history"]:
            counts = Counter(st.session_state["history"])
            df_counts = pd.DataFrame.from_dict(counts, orient="index", columns=["count"]).reindex(MOODS).fillna(0)
            st.bar_chart(df_counts)

        st.markdown("<h3>💡 Suggestion</h3>", unsafe_allow_html=True)
        if mood == "Sad":
            st.info("Here’s an upbeat playlist 🎵 [Play Music](https://www.youtube.com/results?search_query=happy+playlist)")
        elif mood == "Stressed":
            st.info("Take a short break 🌿 [Relaxing Music](https://www.youtube.com/results?search_query=relaxing+music)")
        elif mood == "Happy":
            st.success("Keep smiling ✨ Share your positivity!")
        elif mood == "Angry":
            st.warning("Try calming meditation 🧘 [Meditation](https://www.youtube.com/results?search_query=calming+meditation)")
        elif mood == "Excited":
            st.success("Channel your energy 🚀 [Motivational Playlist](https://www.youtube.com/results?search_query=motivational+music)")
        else:
            st.info("Stay balanced and relaxed ☕")
