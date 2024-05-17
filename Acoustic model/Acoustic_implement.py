import librosa
import numpy as np
import sounddevice as sd
import tensorflow as tf

# Set the path to the pre-trained model .h5 file
MODEL_PATH = "E:/Acoustic/model_1.h5"

# Load the pre-trained audio event detection model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model.compile()

# Set the required audio parameters
SAMPLE_RATE = 22050  # Sample rate in Hz
DURATION = 1  # Length of audio chunks in seconds
CHUNK_SIZE = int(SAMPLE_RATE * DURATION)  # Number of audio samples per chunk

# Define the function for processing audio chunks


def process_audio_chunk(chunk):
    # Convert audio chunk to mono and normalize the amplitude
    mono_chunk = librosa.to_mono(chunk)
    normalized_chunk = librosa.util.normalize(mono_chunk)

    # Extract audio features (e.g., mel spectrogram) for the chunk
    # Replace with your feature extraction function
    features = extract_features(normalized_chunk)

    # Perform inference on the chunk to detect the acoustic event
    # Replace with your model's inference method
    prediction = model.predict(features)

    # Check if the acoustic event is detected
    if prediction == 1:
        print("Warning: Acoustic Event Detected!")

# Define the callback function for audio stream processing


def audio_callback(indata, frames, time, status):
    # Only process audio if there is no status message (no errors)
    if status:
        print('Error:', status)
        return

    # Process the audio stream in chunks
    for i in range(0, len(indata), CHUNK_SIZE):
        chunk = indata[i:i + CHUNK_SIZE]
        process_audio_chunk(chunk)


# Start the microphone input stream
with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE):
    print("Real-time acoustic detection started. Press Ctrl+C to stop.")
    while True:
        pass
