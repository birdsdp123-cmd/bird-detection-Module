import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# ------------------------------
# --- Streamlit Page Setup ---
# ------------------------------
st.set_page_config(
    page_title="Bird Detector Module 🐦",
    page_icon="🐦",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ------------------------------
# --- Custom CSS for Modern UI ---
# ------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1506744038136-46273834b3fb");
        background-size: cover;
        background-attachment: fixed;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 8px 24px;
        font-size:16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🦜🎥 Bird Detterence System - Detection Module ")
st.write("Upload your YOLO model and test images!")

# ------------------------------
# --- Sidebar Controls ---
# ------------------------------
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05
)
use_webcam = st.sidebar.checkbox("Use Webcam", value=False)

# Add image size input
resize_width = st.sidebar.number_input("Resize Image Width (px)", min_value=100, max_value=2000, value=640)
resize_height = st.sidebar.number_input("Resize Image Height (px, 0 to keep aspect ratio)", min_value=0, max_value=2000, value=0)

# ------------------------------
# --- Model Upload ---
# ------------------------------
uploaded_model = st.file_uploader("Upload your YOLO `.pt` model file here", type=["pt"])
model = None
if uploaded_model is not None:
    with open("uploaded_model.pt", "wb") as f:
        f.write(uploaded_model.getbuffer())
    st.success("Model uploaded successfully!")

    @st.cache_resource
    def load_model(path):
        m = torch.load(path)
        m.eval()
        return m

    model = load_model("uploaded_model.pt")
else:
    st.warning("Please upload your `.pt` model to start detection.")
    st.stop()

# ------------------------------
# --- Image Processing Function ---
# ------------------------------
def process_image(image_cv):
    # Resize image if width/height provided
    h, w = image_cv.shape[:2]
    if resize_height == 0:  # Keep aspect ratio
        ratio = resize_width / w
        new_h = int(h * ratio)
        image_cv = cv2.resize(image_cv, (resize_width, new_h))
    else:
        image_cv = cv2.resize(image_cv, (resize_width, resize_height))

    # Inference
    results = model(image_cv)

    # Filter boxes by confidence threshold
    filtered_results = []
    for box, score, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
        if score >= confidence_threshold:
            filtered_results.append((box, score, cls))

    annotated_image = results[0].plot()
    return annotated_image

# ------------------------------
# --- Image Input ---
# ------------------------------
if use_webcam:
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    st.write("Press 'Stop' in sidebar to end webcam.")
    
    while cap.isOpened() and st.sidebar.checkbox("Stop Webcam", value=False) == False:
        ret, frame = cap.read()
        if not ret:
            st.warning("No camera feed detected.")
            break
        annotated_frame = process_image(frame)
        stframe.image(annotated_frame, channels="BGR")
    cap.release()
else:
    uploaded_file = st.file_uploader("Upload an image for bird detection", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        annotated_image = process_image(image_cv)
        st.image(annotated_image, caption="Detected Birds", use_column_width=True)
