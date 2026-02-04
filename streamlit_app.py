import os
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 224
CONF_THRESHOLD = 0.5
IMAGE_PATH = "What-is-Facial-Recognition.webp"

GREEN = (0, 200, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

st.set_page_config(
    page_title="Face Attendance System",
    page_icon="📸",
    layout="wide"
)

# -----------------------------
# LOAD MODEL & LABELS (FIXED FOR DOCKER/KERAS 3)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "face_recognition_mobilenetv2.h5")
labels_path = os.path.join(BASE_DIR, "class_indices.json")

@st.cache_resource
def get_model():
    # Adding compile=False often bypasses the 'pop' error in Keras 3 
    # when loading legacy H5 files.
    return load_model(model_path, compile=False)

try:
    model = get_model()
    with open(labels_path) as f:
        class_indices = json.load(f)
    labels = {v: k for k, v in class_indices.items()}
except Exception as e:
    st.error(f"Error loading model or labels: {e}")
    st.stop()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# HERO SECTION
# -----------------------------
st.markdown(
    """
    <div style="text-align:center; padding:30px">
        <h1 style="font-size:48px">📸 Face Attendance System</h1>
        <p style="font-size:20px; color:gray">AI-powered attendance using MobileNetV2</p>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# SIDEBAR & ATTENDANCE LOGIC
# -----------------------------
menu = st.sidebar.radio("Choose Section", ["🏫 Mark Attendance", "📥 Download Attendance"])

if "attendance" not in st.session_state:
    st.session_state.attendance = []

def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    
    if not any(row["Name"] == name and row["Date"] == date for row in st.session_state.attendance):
        st.session_state.attendance.append({"Name": name, "Date": date, "Time": time})

# -----------------------------
# MARK ATTENDANCE
# -----------------------------
if menu == "🏫 Mark Attendance":
    run = st.checkbox('Start Camera')
    frame_placeholder = st.empty()
    
    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not found or disconnected.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                face = face.astype("float32") / 255.0
                face = np.expand_dims(face, axis=0)

                preds = model.predict(face, verbose=0)
                idx = np.argmax(preds)
                confidence = preds[0][idx]

                if confidence > CONF_THRESHOLD:
                    name = labels[idx]
                    color = GREEN
                    mark_attendance(name)
                    label = f"{name} {confidence:.2f}"
                else:
                    color = RED
                    label = f"UNKNOWN {confidence:.2f}"

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")
        
        cap.release()

# -----------------------------
# DOWNLOAD ATTENDANCE
# -----------------------------
elif menu == "📥 Download Attendance":
    if not st.session_state.attendance:
        st.warning("No attendance recorded.")
    else:
        df = pd.DataFrame(st.session_state.attendance)
        st.dataframe(df, use_container_width=True)
        st.download_button("⬇️ Download CSV", df.to_csv(index=False), "attendance.csv")
