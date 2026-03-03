# fast_train_all_modalities.py
import os, random
import numpy as np
import tensorflow as tf
import librosa, cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, LSTM, TimeDistributed, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths
# -----------------------------
IMAGE_TRAIN = "image_dataset/train"
IMAGE_VAL = "image_dataset/val"
IMAGE_TEST = "image_dataset/test"

AUDIO_TRAIN = "audio_dataset/train"
AUDIO_VAL = "audio_dataset/val"
AUDIO_TEST = "audio_dataset/test"

VIDEO_PATH = "video_dataset"

# -----------------------------
# IMAGE MODEL FAST
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 2   # only 2 epochs for fast demo

print(">> Fast Image Training ...")
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Sample smaller dataset for speed (10% randomly)
def flow_subset(directory, target_size, batch_size, subset_ratio=0.1):
    all_files = []
    classes = os.listdir(directory)
    for cls in classes:
        cls_folder = os.path.join(directory, cls)
        files = os.listdir(cls_folder)
        k = max(1, int(len(files)*subset_ratio))
        sample_files = random.sample(files, k)
        for f in sample_files:
            all_files.append(os.path.join(cls_folder, f))
    # Use flow_from_dataframe trick
    import pandas as pd
    df = pd.DataFrame({'filename': all_files, 'class': [f.split(os.sep)[-2] for f in all_files]})
    return train_datagen.flow_from_dataframe(df, x_col='filename', y_col='class', target_size=target_size, class_mode='binary', batch_size=batch_size)

train_gen = flow_subset(IMAGE_TRAIN, (IMG_SIZE, IMG_SIZE), BATCH_SIZE)
val_gen = flow_subset(IMAGE_VAL, (IMG_SIZE, IMG_SIZE), BATCH_SIZE)
test_gen = flow_subset(IMAGE_TEST, (IMG_SIZE, IMG_SIZE), BATCH_SIZE)

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # freeze layers for speed
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)
image_model = Model(base_model.input, output)
image_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

history_img = image_model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
img_acc = image_model.evaluate(test_gen)
print(f"Image Test Accuracy: {img_acc[1]*100:.2f}%")

# -----------------------------
# AUDIO MODEL FAST
# -----------------------------
def create_spec(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    y = librosa.util.fix_length(y, size=sr*3)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    return librosa.power_to_db(mel, ref=np.max)

def load_audio_subset(path, subset_ratio=0.2):
    X, y = [], []
    for label, cls in enumerate(['real','fake']):
        folder = os.path.join(path, cls)
        files = os.listdir(folder)
        k = max(1, int(len(files)*subset_ratio))
        files = random.sample(files, k)
        for f in files:
            try:
                mel = create_spec(os.path.join(folder, f))
                mel = cv2.resize(mel, (64,64))
                X.append(mel)
                y.append(label)
            except: continue
    return np.array(X)[..., np.newaxis]/255.0, np.array(y)

print(">> Fast Audio Training ...")
X_train_audio, y_train_audio = load_audio_subset(AUDIO_TRAIN)
X_val_audio, y_val_audio = load_audio_subset(AUDIO_VAL)
X_test_audio, y_test_audio = load_audio_subset(AUDIO_TEST)

audio_model = Sequential([
    Conv2D(16,(3,3),activation='relu', input_shape=(64,64,1)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
audio_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
history_audio = audio_model.fit(X_train_audio, y_train_audio, validation_data=(X_val_audio, y_val_audio), epochs=EPOCHS, batch_size=16)
audio_acc = audio_model.evaluate(X_test_audio, y_test_audio)
print(f"Audio Test Accuracy: {audio_acc[1]*100:.2f}%")

# -----------------------------
# VIDEO MODEL FAST
# -----------------------------
FRAME_SIZE = 112
FRAMES_PER_VIDEO = 5

def extract_frames(video_file, max_frames=FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(video_file)
    frames=[]
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total==0: return frames
    step = max(total//max_frames,1)
    idx=0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx%step==0:
            frame = cv2.resize(frame,(FRAME_SIZE,FRAME_SIZE))
            frame = frame.astype('float32')/255.0
            frames.append(frame)
            if len(frames)>=max_frames: break
        idx+=1
    cap.release()
    while len(frames)<max_frames:
        frames.append(np.zeros((FRAME_SIZE,FRAME_SIZE,3)))
    return np.array(frames)

def load_video_subset(path, subset_ratio=0.2):
    X,y=[],[]
    for label, cls in enumerate(['Real','Fake']):
        folder=os.path.join(path, cls)
        files=os.listdir(folder)
        k=max(1,int(len(files)*subset_ratio))
        files=random.sample(files,k)
        for f in files:
            try:
                X.append(extract_frames(os.path.join(folder,f)))
                y.append(label)
            except: continue
    return np.array(X), np.array(y)

print(">> Fast Video Training ...")
X_video, y_video = load_video_subset(VIDEO_PATH)
X_train_v, X_temp, y_train_v, y_temp = train_test_split(X_video, y_video, test_size=0.3, random_state=42, stratify=y_video)
X_val_v, X_test_v, y_val_v, y_test_v = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

cnn_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(FRAME_SIZE,FRAME_SIZE,3))
cnn_base.trainable=False
cnn_out=GlobalAveragePooling2D()(cnn_base.output)
cnn_model = Model(cnn_base.input, cnn_out)

video_model = Sequential([
    TimeDistributed(cnn_model, input_shape=(FRAMES_PER_VIDEO,FRAME_SIZE,FRAME_SIZE,3)),
    LSTM(32),
    Dropout(0.3),
    Dense(1,activation='sigmoid')
])
video_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
history_video = video_model.fit(X_train_v, y_train_v, validation_data=(X_val_v, y_val_v), epochs=EPOCHS, batch_size=2)
video_acc = video_model.evaluate(X_test_v, y_test_v)
print(f"Video Test Accuracy: {video_acc[1]*100:.2f}%")

# -----------------------------
# Save Models
# -----------------------------
os.makedirs("models", exist_ok=True)
image_model.save("models/image_model.h5")
audio_model.save("models/audio_model.h5")
video_model.save("models/video_model.h5")
print("✅ Fast training done, models saved in ./models")