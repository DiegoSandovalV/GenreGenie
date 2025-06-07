import os
import uuid
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.initializers import HeNormal
import librosa
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr
import yt_dlp
from pydub import AudioSegment

# Constants
GENRE_LABELS = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]
SAMPLE_RATE = 22050
SEGMENT_DURATION = 3
IMAGE_SIZE = (130, 128)
MODEL_PATH = '/home/diego/GenreGenie/models/best_model.weights.h5'
TEMP_IMG_PATH = "/home/diego/GenreGenie/app/test/gradio_segment.png"

# Model definition
def get_model(input_shape=(130, 128)):
    input_shape = (input_shape[0], input_shape[1], 3)
    model = Sequential([
        Conv2D(32, kernel_size=3, padding='same', activation='relu', kernel_initializer=HeNormal(), input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer=HeNormal()),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(128, kernel_size=3, padding='same', activation='relu', kernel_initializer=HeNormal()),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu', kernel_initializer=HeNormal()),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    return model

def load_trained_model():
    model = get_model(IMAGE_SIZE)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    model.load_weights(MODEL_PATH)
    return model

# Spectrogram generation
def generate_mel_spectrogram(y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    db_mel = librosa.power_to_db(mel, ref=np.max)

    if db_mel.shape[1] > 130:
        db_mel = db_mel[:, :130]
    elif db_mel.shape[1] < 130:
        pad_width = 130 - db_mel.shape[1]
        db_mel = np.pad(db_mel, ((0, 0), (0, pad_width)), mode='constant')

    return db_mel

def save_spectrogram_image(spec, save_path):
    spec = np.flipud(spec)
    plt.imsave(save_path, spec, cmap='viridis', origin='lower')
    print(f"[DEBUG] Spectrogram image saved to: {save_path}")
    return save_path

# Image loader
def load_image(path, size):
    img = Image.open(path).convert("RGB")
    img = img.resize(size[::-1], resample=getattr(Image, 'Resampling', Image).BILINEAR)
    return np.expand_dims(np.array(img, dtype=np.float32), axis=0)

# Model prediction
def predict(model, image_path):
    img = load_image(image_path, IMAGE_SIZE)
    preds = model.predict(img)
    return GENRE_LABELS[np.argmax(preds)]

# YouTube search downloader with clip trimming
def download_youtube_audio(search_query, output_dir="/tmp", clip_duration=60):
    audio_id = str(uuid.uuid4())
    temp_mp4 = os.path.join(output_dir, f"{audio_id}.mp4")
    temp_wav = os.path.join(output_dir, f"{audio_id}.wav")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': temp_mp4,
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"[DEBUG] Searching and downloading first result for: {search_query}")
        ydl.download([f'ytsearch1:{search_query}'])

    audio = AudioSegment.from_file(temp_mp4)

    # Calculate start and end times for trimming (1 minute)
    duration_ms = len(audio)
    clip_duration_ms = clip_duration * 1000
    start_ms = max((duration_ms - clip_duration_ms) // 2, 0)
    end_ms = min(start_ms + clip_duration_ms, duration_ms)

    trimmed_audio = audio[start_ms:end_ms]
    trimmed_audio.export(temp_wav, format="wav")
    print(f"[DEBUG] Trimmed audio saved to: {temp_wav}")

    os.remove(temp_mp4)

    return temp_wav

# Main prediction logic
model = load_trained_model()

def predict_genre(search_query, audio_file):
    audio_path = None
    cleanup = False

    if search_query and search_query.strip():
        audio_path = download_youtube_audio(search_query, clip_duration=60)
        cleanup = True
    elif audio_file:
        audio_path = audio_file
        cleanup = False
    else:
        return "Please provide a search query or upload an audio file.", None

    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        segment_len = int(SAMPLE_RATE * SEGMENT_DURATION)
        num_segments = min(len(y) // segment_len, 20)

        if num_segments == 0:
            raise ValueError("Audio is too short for a full segment")

        predictions = []

        for i in range(num_segments):
            start = i * segment_len
            end = start + segment_len
            segment = y[start:end]

            spec = generate_mel_spectrogram(segment, sr)
            temp_path = f"/tmp/gradio_segment_{i}.png"
            save_spectrogram_image(spec, temp_path)
            pred = predict(model, temp_path)
            print(f"[DEBUG] Segment {i+1}: Predicted '{pred}'")
            predictions.append(pred)

        freq = {}
        for genre in predictions:
            freq[genre] = freq.get(genre, 0) + 1

        most_common = max(freq.items(), key=lambda x: x[1])[0]
        print(f"[DEBUG] Final Prediction (Most Frequent): {most_common}")

        return most_common, audio_path

    except Exception as e:
        return f"Error: {str(e)}", None

    finally:
        if cleanup and audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"[DEBUG] Cleaned up temporary audio file: {audio_path}")


gr.Interface(
    fn=predict_genre,
    inputs=[
        gr.Textbox(label="Search YouTube for (leave blank if uploading audio)"),
        gr.Audio(type="filepath", label="Upload audio clip (leave blank if using YouTube search)")
    ],
    outputs=[
        gr.Textbox(label="Predicted Genre"),
        gr.Audio(label="Trimmed Audio Preview")
    ],
    title="🎵 Genre Genie 🎵",
    description="Enter a YouTube search query OR upload an audio clip (at least 3 seconds) to predict the genre.\n"
                "For YouTube, it will process 1 minute from the middle of the first video result."
).launch()
