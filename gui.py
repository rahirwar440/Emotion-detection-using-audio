import tkinter as tk
from tkinter import filedialog, messagebox
import sounddevice as sd
from scipy.io.wavfile import write
import pickle
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import soundfile
import librosa

# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(y = X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

# Emotion and observed_emotions dictionary setup here

def upload_audio_file():
    global audio_file_path
    filename = filedialog.askopenfilename()  # show an "Open" dialog box and return the path to the selected file
    if filename:  # If a file was selected
        audio_file_path = filename
        print(f"Audio file {filename} loaded")
        messagebox.showinfo("File Loaded", f"Audio file {filename} loaded successfully!")

def detect_emotion():
    if os.path.exists(audio_file_path):
        loaded_model = pickle.load(open('modelForPrediction1.sav', 'rb'))  # Load your model
        feature = extract_feature(audio_file_path, mfcc=True, chroma=True, mel=True)  # Extract features
        feature = feature.reshape(1, -1)
        prediction = loaded_model.predict(feature)
        print(prediction)
        emotion_label.config(text=f"Emotion: {prediction[0]}")
    else:
        messagebox.showwarning("Error", "Please record or upload an audio clip first!")

# GUI setup remains the same

# Set up the GUI
root = tk.Tk()
root.title("Emotion Detection from Audio")

frame = tk.Frame(root)
frame.pack(pady=20)

upload_btn = tk.Button(frame, text="Upload Audio File", command=upload_audio_file)
upload_btn.pack(side=tk.LEFT, padx=10)

detect_btn = tk.Button(frame, text="Detect Emotion", command=detect_emotion)
detect_btn.pack(side=tk.LEFT, padx=10)

emotion_label = tk.Label(root, text="Emotion: ")
emotion_label.pack(pady=20)

root.mainloop()

audio_file_path = ''  # Initialize the variable to store audio file path
