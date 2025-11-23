import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile

st.title("YOLO Model Image/Video Testing App")

st.write("Upload your YOLO .pt model file first.")

# -------------------------
# Step 1: User uploads model
# -------------------------
model_file = st.file_uploader("Upload YOLO model (.pt)", type=["pt"])

if model_file is not None:
    # Save model to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(model_file.read())
        model_path = tmp.name

    st.success("Model uploaded successfully! Loading model...")

    # Load YOLO model
    try:
        model = torch.load(model_path, map_location=torch.device("cpu"))
        model.eval()
        st.success("YOLO model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    st.write("Now upload an image or video.")

    # -------------------------
    # Step 2: User uploads media
    # -------------------------
    media_file = st.file_uploader("Upload Image or Video", 
                                  type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

    if media_file is not None:
        # -------------------------
        # IMAGE PROCESSING
        # -------------------------
        if media_file.type.startswith("image"):
            image = Image.open(media_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            img_np = np.array(image)

            # YOLO inference
            try:
                results = model(img_np)
                annotated_img = results.render()[0]

                st.image(annotated_img, caption="YOLO Detection Result", use_column_width=True)
            except Exception as e:
                st.error(f"Model inference failed: {e}")

        # -------------------------
        # VIDEO PROCESSING
        # -------------------------
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(media_file.read())
            video_path = tfile.name

            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()

            st.write("Processing video...")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                try:
                    results = model(frame_rgb)
                    annotated_frame = results.render()[0]
                except:
                    annotated_frame = frame_rgb

                stframe.image(annotated_frame, channels="RGB")

            cap.release()
