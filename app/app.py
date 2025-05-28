import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.initializers import HeNormal
import librosa
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr

# Constants
GENRE_LABELS = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]
SAMPLE_RATE = 22050
SEGMENT_DURATION = 3
IMAGE_SIZE = (130, 128)  
MODEL_PATH = '/home/diego/GenreGenie/models/best_model.weights.h5'
TEMP_IMG_PATH = "/home/diego/GenreGenie/test/gradio_segment3.png"

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


def generate_mel_spectrogram(y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    db_mel = librosa.power_to_db(mel, ref=np.max)

    if db_mel.shape[1] > 130:
        db_mel = db_mel[:, :130]
    elif db_mel.shape[1] < 130:
        pad_width = 130 - db_mel.shape[1]
        db_mel = np.pad(db_mel, ((0, 0), (0, pad_width)), mode='constant')

    print(f"[DEBUG] Spectrogram shape after resize: {db_mel.shape}")
    return db_mel

def save_spectrogram_image(spec, save_path):
    spec = np.flipud(spec)
    plt.imsave(save_path, spec, cmap='viridis', origin='lower')
    print(f"[DEBUG] Spectrogram image saved to: {save_path}")
    return save_path

def extract_first_segment_and_save_image(audio_path, save_path=TEMP_IMG_PATH):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    segment_len = int(SAMPLE_RATE * SEGMENT_DURATION)

    if len(y) < segment_len:
        raise ValueError("Audio is too short for a full segment")

    segment = y[:segment_len]
    spec = generate_mel_spectrogram(segment, sr)
    save_spectrogram_image(spec, save_path)
    return save_path


def load_image(path, size):
    img = Image.open(path).convert("RGB")
    img = img.resize(size[::-1], resample=getattr(Image, 'Resampling', Image).BILINEAR)
    return np.expand_dims(np.array(img, dtype=np.float32), axis=0)

def predict(model, image_path):
    img = load_image(image_path, IMAGE_SIZE)
    preds = model.predict(img)
    return GENRE_LABELS[np.argmax(preds)]


model = load_trained_model()

def predict_genre(audio_path):
    print(f"[DEBUG] Received audio path: {audio_path}")
    try:
        extract_first_segment_and_save_image(audio_path, TEMP_IMG_PATH)
        return predict(model, TEMP_IMG_PATH)
    except Exception as e:
        return f"Error: {str(e)}"


gr.Interface(
    fn=predict_genre,
    inputs=gr.Audio(type="filepath", label="Upload an audio clip"),
    outputs=gr.Textbox(label="Predicted Genre"),
    title="🎵 Music Genre Classifier",
    description="Upload a short audio clip (at least 3 seconds) to predict the genre using mel spectrograms."
).launch()
