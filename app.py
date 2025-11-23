import streamlit as st
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv
from torch.nn.modules.container import Sequential
import tempfile
import numpy as np
from PIL import Image
import cv2

# -----------------------------
# Safe loading for YOLO models
# -----------------------------
torch.serialization.add_safe_globals([DetectionModel, Conv, Sequential])

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="Bird Detector Module",
    page_icon="🐦",
    layout="centered"
)

st.title("🦜 Bird Detection Module")
st.write("Upload your YOLO model file first, then choose an image or video to test.")

# -----------------------------
# Step 1: Upload YOLO Model
# -----------------------------
st.header("1️⃣ Upload YOLO Model (.pt)")
uploaded_model = st.file_uploader("Upload YOLO .pt model", type=["pt"])

if uploaded_model is None:
    st.info("Please upload your YOLO model to continue.")
    st.stop()

# Save model temporarily
with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
    tmp.write(uploaded_model.read())
    model_path = tmp.name

# Load model
try:
    model = YOLO(model_path)
    st.success("🎉 Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -----------------------------
# Step 2: Choose Input Type
# -----------------------------
st.header("2️⃣ Choose input: Image or Video")
input_type = st.radio("Select input type:", ["Image", "Video"])
confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

# -----------------------------
# IMAGE PROCESSING
# -----------------------------
if input_type == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Convert to BGR
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Run YOLO inference
        results = model(img_cv, conf=confidence)
        annotated = results[0].plot()

        st.image(annotated, caption="Detection Result", use_column_width=True)

# -----------------------------
# VIDEO PROCESSING
# -----------------------------
else:
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        st.video(uploaded_video)

        # Save video temporarily
        temp_vid = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_vid.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_vid.name)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO inference
            results = model(frame, conf=confidence)
            annotated_frame = results[0].plot()

            # Display frame
            stframe.image(annotated_frame, channels="BGR")

        cap.release()
        st.success("Video processing completed!")
