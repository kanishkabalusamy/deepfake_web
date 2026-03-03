import os
import numpy as np
import cv2
import librosa
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# -----------------------------
# Flask App Setup
# -----------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
IMAGE_FOLDER = os.path.join(UPLOAD_FOLDER, "images")
VIDEO_FOLDER = os.path.join(UPLOAD_FOLDER, "videos")
AUDIO_FOLDER = os.path.join(UPLOAD_FOLDER, "audios")
SPECTROGRAM_FOLDER = "static/spectrograms"

os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(SPECTROGRAM_FOLDER, exist_ok=True)

# -----------------------------
# Load Models
# -----------------------------
image_model = load_model("models/image_model.h5")
video_model = load_model("models/video_model.h5")
audio_model = load_model("models/audio_model.h5")

IMG_SIZE = 224  # EfficientNet input size
FRAMES_PER_VIDEO = 20

# -----------------------------
# Utility Functions
# -----------------------------
def predict_image(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img.astype("float32"))
    img = np.expand_dims(img, axis=0)
    pred = image_model.predict(img)[0][0]
    return float(pred)

def extract_video_frames(video_path, max_frames=FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return frames
    step = max(total // max_frames, 1)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = preprocess_input(frame.astype("float32"))
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames

def predict_video(video_path):
    frames = extract_video_frames(video_path)
    if len(frames) == 0:
        return None
    frames = np.array(frames)
    preds = video_model.predict(frames)
    avg_pred = float(np.mean(preds))
    return avg_pred

def create_mel_spectrogram(audio_path, output_img_path):
    y, sr = librosa.load(audio_path, sr=16000)
    target_len = sr * 3
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    else:
        y = y[:target_len]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    plt.figure(figsize=(3, 3))
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.imshow(mel_db, aspect="auto", origin="lower")
    plt.savefig(output_img_path, bbox_inches="tight", pad_inches=0)
    plt.close()

def predict_audio(audio_path):
    spec_path = os.path.join(SPECTROGRAM_FOLDER, "temp_spec.png")
    create_mel_spectrogram(audio_path, spec_path)
    img = cv2.imread(spec_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    pred = audio_model.predict(img)[0][0]
    return float(pred), spec_path

def label_from_pred(pred):
    if pred >= 0.5:
        return "FAKE", pred
    else:
        return "REAL", 1 - pred

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/image", methods=["GET","POST"])
def image_page():
    if request.method=="POST":
        file = request.files["file"]
        if not file.filename:
            return render_template("image.html", error="No file selected")
        filename = secure_filename(file.filename)
        filepath = os.path.join(IMAGE_FOLDER, filename)
        file.save(filepath)
        pred = predict_image(filepath)
        label, conf = label_from_pred(pred)
        return render_template("image.html",
                               label=label,
                               conf_score=conf*100,
                               filename=f"uploads/images/{filename}",
                               confidence=f"{conf*100:.2f}%")
    return render_template("image.html")

@app.route("/video", methods=["GET","POST"])
def video_page():
    if request.method=="POST":
        file = request.files["file"]
        if not file.filename:
            return render_template("video.html", error="No file selected")
        filename = secure_filename(file.filename)
        filepath = os.path.join(VIDEO_FOLDER, filename)
        file.save(filepath)
        pred = predict_video(filepath)
        if pred is None:
            return render_template("video.html", error="Video could not be processed")
        label, conf = label_from_pred(pred)
        return render_template("video.html",
                               label=label,
                               conf_score=conf*100,
                               filename=f"uploads/videos/{filename}",
                               confidence=f"{conf*100:.2f}%")
    return render_template("video.html")

@app.route("/audio", methods=["GET","POST"])
def audio_page():
    if request.method=="POST":
        file = request.files["file"]
        if not file.filename:
            return render_template("audio.html", error="No file selected")
        filename = secure_filename(file.filename)
        filepath = os.path.join(AUDIO_FOLDER, filename)
        file.save(filepath)
        pred, spec_path = predict_audio(filepath)
        label, conf = label_from_pred(pred)
        return render_template("audio.html",
                               label=label,
                               conf_score=conf*100,
                               filename=f"uploads/audios/{filename}",
                               spectrogram=spec_path,
                               confidence=f"{conf*100:.2f}%")
    return render_template("audio.html")

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)