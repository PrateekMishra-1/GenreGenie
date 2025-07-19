import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from matplotlib import pyplot
from tensorflow.image import resize
import soundfile as sf
import io

# Cache the model loading
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model("Trained_model.h5")
    return model

# Load and preprocess audio data
def load_and_preprocess_data(file_stream, target_shape=(150, 150)):
    data = []
    
    # Read the uploaded file as binary and decode
    audio_data, sample_rate = sf.read(io.BytesIO(file_stream.read()))
    
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap_duration * sample_rate)
    
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
    
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        
        if len(chunk) < chunk_samples:
            padding = np.zeros(chunk_samples - len(chunk))
            chunk = np.concatenate((chunk, padding))
        
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    return np.array(data)

# TensorFlow Model Prediction
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_elements = unique_elements[counts == np.max(counts)]
    return max_elements[0]

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Home Page
if app_mode == "Home":
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #181646;
            color: white;
        }
        h2, h3 {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## Welcome to the,\n## Music Genre Classification System! 🎶🎧")
    st.image("music_genre_home.png", use_container_width=True)

    st.markdown("""
**Our goal is to help in identifying music genres from audio tracks efficiently. Upload an audio file, and our system will analyze it to detect its genre. Discover the power of AI in music analysis!**

### How It Works
1. **Upload Audio:** Go to the **Genre Classification** page and upload an audio file.
2. **Analysis:** Our system will process the audio using advanced algorithms to classify it into one of the predefined genres.
3. **Results:** View the predicted genre along with related information.

### Why Choose Us?
- **Accuracy:** Our system leverages state-of-the-art deep learning models for accurate genre prediction.
- **User-Friendly:** Simple and intuitive interface for a smooth user experience.
- **Fast and Efficient:** Get results quickly, enabling faster music categorization and exploration.

### Get Started
Click on the **Genre Classification** page in the sidebar to upload an audio file and explore the magic of our Music Genre Classification System!

### About Us
Learn more about the project, our team, and our mission on the **About** page.
""")

# About Project Page
elif app_mode == "About Project":
    st.markdown("""
### About Project
Music experts have been trying for a long time to understand sound and what differentiates one song from another. This project aims to do just that using AI.

### About Dataset
- **genres original** – 10 genres, 100 audio files each (GTZAN dataset)
- **Genres:** blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- **images original** – Mel spectrograms for visual input to CNNs
- **CSV files** – extracted audio features for training ML models
""")

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    test_audio = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
    
    if test_audio is not None:
        st.audio(test_audio)

        if st.button("Predict"):
            with st.spinner("Please wait..."):
                X_test = load_and_preprocess_data(test_audio)
                result_index = model_prediction(X_test)
                st.balloons()

                labels = ['blues', 'classical', 'country', 'disco', 'hiphop',
                          'jazz', 'metal', 'pop', 'reggae', 'rock']
                st.markdown(f"**:blue[Model Prediction:] It's a :red[{labels[result_index]}] music**")
