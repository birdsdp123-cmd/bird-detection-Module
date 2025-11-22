import os
import base64
import requests
import torch
import streamlit as st
from PIL import Image
import cv2
import numpy as np

# Function to upload files to GitHub using GitHub API
def upload_file_to_github(file_path, repo_owner, repo_name, file_name, commit_message, github_token):
    """Upload file to GitHub repository using GitHub API"""
    with open(file_path, "rb") as f:
        content = f.read()
    
    # Encode the file content to base64
    encoded_content = base64.b64encode(content).decode('utf-8')
    
    # Define GitHub API URL for uploading files
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_name}"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Create the data payload for uploading
    data = {
        "message": commit_message,
        "content": encoded_content,
        "branch": "main"  # You can change this to your target branch
    }
    
    # Make the request to upload the file
    response = requests.put(url, json=data, headers=headers)
    
    if response.status_code == 201:
        st.success(f"Successfully uploaded {file_name} to GitHub!")
    else:
        st.error(f"Failed to upload file. Error: {response.status_code} - {response.text}")

# Streamlit interface setup
st.title("YOLO Model and Image/Video Upload to GitHub")

# User inputs for GitHub details
repo_owner = "your-github-username"  # Your GitHub username
repo_name = "your-repo-name"  # Your GitHub repository name
github_token = "your-github-token"  # Your GitHub Personal Access Token (PAT)

# Step 1: Upload the YOLO Model
uploaded_model = st.file_uploader("Upload your YOLO model `.pt` file", type=["pt"])

if uploaded_model:
    model_path = os.path.join("temp", uploaded_model.name)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the uploaded model temporarily
    with open(model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
    
    # Upload model file to GitHub
    commit_message = f"Upload YOLO model {uploaded_model.name} via Streamlit"
    upload_file_to_github(model_path, repo_owner, repo_name, uploaded_model.name, commit_message, github_token)
    
    st.success(f"YOLO model '{uploaded_model.name}' uploaded successfully!")

    # Load the model (YOLOv5 example)
    model = torch.load(model_path)
    model.eval()

    # Step 2: Upload an Image or Video for testing
    uploaded_file = st.file_uploader("Upload an image or video for testing", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
    
    if uploaded_file:
        # Save the uploaded image/video temporarily
        temp_file_path = os.path.join("temp", uploaded_file.name)
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Upload the file to GitHub
        commit_message = f"Upload {uploaded_file.name} for testing via Streamlit"
        upload_file_to_github(temp_file_path, repo_owner, repo_name, uploaded_file.name, commit_message, github_token)

        # Display the uploaded file on Streamlit
        if uploaded_file.name.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Display image
            image = Image.open(temp_file_path)
            st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
            
            # Optionally: Run the model on the image (YOLOv5 inference example)
            image_cv = np.array(image)
            results = model(image_cv)  # Run YOLOv5 inference on the image
            annotated_image = results.render()[0]  # Draw bounding boxes on the image
            st.image(annotated_image, caption="Detected Image", use_column_width=True)
        
        elif uploaded_file.name.lower().endswith(('.mp4', '.avi', '.mov')):
            # Display video
            cap = cv2.VideoCapture(temp_file_path)
            if not cap.isOpened():
                st.error("Error: Couldn't open video file.")
            else:
                stframe = st.empty()
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, channels="RGB")
                cap.release()
else:
    st.warning("Please upload your YOLO model file first to start the process.")
