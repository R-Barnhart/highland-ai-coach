# Highland Games AI Lab - Streamlit App (Fixed for MediaPipe Tasks API)

# Highland Games AI Lab - Streamlit App (Fixed for MediaPipe Tasks API)
# Highland Games AI Lab - Streamlit App (Fixed for MediaPipe Tasks API)

import streamlit as st
import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmark
from mediapipe.tasks.python.core.base_options import BaseOptions

# --- 1. FRONTEND CONFIGURATION ---
st.set_page_config(page_title="Highland Games AI Lab", layout="wide")
st.title("Highland Games AI Performance Lab")
st.write("""
Upload your throwing video and analyze the biomechanics using AI-powered pose estimation.
""")

# --- 2. UPLOAD VIDEO ---
uploaded_file = st.file_uploader("Upload Your Throw", type=["mp4", "mov"])

# --- 3. INITIALIZE MEDIAPIPE POSE LANDMARKER ---
MODEL_PATH = "pose_landmarker_lite.task"  # Ensure this file exists in project folder

base_options = BaseOptions(model_asset_path=MODEL_PATH)
pose_options = PoseLandmarker.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

pose_landmarker = PoseLandmarker.create_from_options(pose_options)

# --- 4. PROCESS VIDEO ---
if uploaded_file:
    # Load video with OpenCV
    cap = cv2.VideoCapture(uploaded_file.name)

    stframe = st.empty()  # Placeholder for video frames
    peak_angle = 0  # Initialize peak hip angle

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- PoseLandmarker detection ---
        result = pose_landmarker.detect_for_video(
            frame_rgb,
            timestamp_ms=int(cap.get(cv2.CAP_PROP_POS_MSEC))
        )

        # Draw landmarks
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            for connection in PoseLandmark.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                h, w, _ = frame.shape
                start_coord = (int(start.x * w), int(start.y * h))
                end_coord = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_coord, end_coord, (0, 255, 0), 2)

            # Example: Calculate peak hip angle
            left_hip = landmarks[PoseLandmark.LEFT_HIP]
            left_knee = landmarks[PoseLandmark.LEFT_KNEE]
            left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]

            # Compute angle
            v1 = np.array([left_shoulder.x - left_hip.x, left_shoulder.y - left_hip.y])
            v2 = np.array([left_knee.x - left_hip.x, left_knee.y - left_hip.y])
            cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            angle_deg = np.degrees(angle)
            peak_angle = max(peak_angle, angle_deg)

        # Display frame in Streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

    # --- 5. DISPLAY RESULTS ---
    st.subheader("Results")
    st.metric("Peak Hip Angle", f"{int(peak_angle)}°")

