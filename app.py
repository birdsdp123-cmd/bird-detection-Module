import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.auth.transport.requests import Request

# Function to authenticate with Google Drive
def authenticate_google_drive(credentials_file):
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_file, scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return creds

# Authenticate using the Google Drive credentials
credentials_path = "Download.Json.json"  # Path to your 'Download.Json.json' file
creds = authenticate_google_drive(credentials_path)

# Build the Drive API service
service = build('drive', 'v3', credentials=creds)

# Function to download the YOLO model from Google Drive
def download_file_from_drive(file_id, destination_path):
    request = service.files().get_media(fileId=file_id)
    fh = open(destination_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    fh.close()

# List files from Google Drive to get the model ID
results = service.files().list(pageSize=10, fields="files(id, name)").execute()
items = results.get('files', [])

# Display the files in your Drive
if not items:
    st.write('No files found.')
else:
    st.write('Files in your Google Drive:')
    for item in items:
        st.write(f'{item["name"]} ({item["id"]})')
        if item["name"] == "train2_best.pt":  # Match the model file name
            model_file_id = item["id"]
            download_file_from_drive(model_file_id, "train2_best.pt")
            st.success(f"Downloaded model: {item['name']}")

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
st.write("Test images or webcam for bird detection!")

# ------------------------------
# --- Sidebar Controls ---
# ------------------------------
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05
)
use_webcam = st.sidebar.checkbox("Use Webcam", value=False)

resize_width = st.sidebar.number_input("Resize Image Width (px)", min_value=100, max_value=2000, value=640)
resize_height = st.sidebar.number_input("Resize Image Height (px, 0 to keep aspect ratio)", min_value=0, max_value=2000, value=0)

# Load the model after downloading it from Google Drive
@st.cache_resource
def load_model(path):
    model = torch.load(path)
    model.eval()
    return model

model = load_model("train2_best.pt")

# ------------------------------
# --- Image Processing Function ---
# ------------------------------
def process_image(image_cv):
    h, w = image_cv.shape[:2]
    if resize_height == 0:
        ratio = resize_width / w
        new_h = int(h * ratio)
        image_cv = cv2.resize(image_cv, (resize_width, new_h))
    else:
        image_cv = cv2.resize(image_cv, (resize_width, resize_height))

    results = model(image_cv)

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


