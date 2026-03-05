import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker
from mediapipe.tasks.python.vision import PoseLandmarkerOptions
from mediapipe.tasks.python.vision import PoseLandmark
from mediapipe.tasks.python.core.base_options import BaseOptions

# -----------------------------
# 1. STREAMLIT PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Highland Games AI Lab", layout="wide")

st.title("Highland Games AI Lab")
st.write("Upload a throwing video and the AI will analyze hip drive and posture.")

# -----------------------------
# 2. LOAD POSE MODEL
# -----------------------------
MODEL_PATH = "pose_landmarker_lite.task"

base_options = BaseOptions(model_asset_path=MODEL_PATH)

options = PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

pose_landmarker = PoseLandmarker.create_from_options(options)

# -----------------------------
# 3. VIDEO UPLOADER
# -----------------------------
uploaded_file = st.file_uploader("Upload Throw Video", type=["mp4","mov","avi"])

if uploaded_file is not None:

    st.success("Video uploaded!")

    # Save uploaded video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    video_path = tfile.name

    # Display original video
    st.subheader("Original Video")
    st.video(video_path)

    # -----------------------------
    # 4. VIDEO PROCESSING
    # -----------------------------
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    hip_angles = []

    st.subheader("Processing Frames...")

    progress = st.progress(0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        results = pose_landmarker.detect_for_video(mp_image, timestamp)

        if results.pose_landmarks:

            landmarks = results.pose_landmarks[0]

            left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
            left_hip = landmarks[PoseLandmark.LEFT_HIP]
            left_knee = landmarks[PoseLandmark.LEFT_KNEE]

            # Convert to numpy points
            shoulder = np.array([left_shoulder.x, left_shoulder.y])
            hip = np.array([left_hip.x, left_hip.y])
            knee = np.array([left_knee.x, left_knee.y])

            # Calculate hip angle
            a = shoulder - hip
            b = knee - hip

            cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            angle = np.degrees(np.arccos(cosine_angle))

            hip_angles.append(angle)

        progress.progress(frame_count / total_frames)

    cap.release()

    # -----------------------------
    # 5. RESULTS
    # -----------------------------
    st.subheader("Analysis Results")

    if len(hip_angles) > 0:

        peak_angle = max(hip_angles)

        st.metric("Peak Hip Drive Angle", f"{peak_angle:.1f}°")

        st.line_chart(hip_angles)

        if peak_angle < 150:
            st.warning("Limited hip extension. Drive harder through the hips.")
        else:
            st.success("Excellent hip extension!")

    else:
        st.error("No pose detected. Try a clearer video.")



