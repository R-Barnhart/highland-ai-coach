import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker
import openai
import time

# -----------------------------
# 1. STREAMLIT PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Highland Games AI Lab", layout="wide")
st.title("🏴 Highland Games AI Performance Lab")

# -----------------------------
# 2. OPENAI SETUP
# -----------------------------
# Set your OpenAI API key here or via environment variable
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

# -----------------------------
# 3. EVENT SELECTION
# -----------------------------
event = st.selectbox(
    "Select Event",
    ["WOB", "WFD", "Hammer", "Stones", "Caber", "Sheaf"]
)

# -----------------------------
# 4. VIDEO UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload Your Throw Video", type=["mp4", "mov"])
if not uploaded_file:
    st.info("Please upload a video to analyze.")
    st.stop()

# Save uploaded video to temp file
tfile = tempfile.NamedTemporaryFile(delete=False)
tfile.write(uploaded_file.read())
video_path = tfile.name

# -----------------------------
# 5. INITIALIZE MEDIAPIPE POSELANDMARKER
# -----------------------------
MODEL_PATH = "pose_landmarker_lite.task"  # Must exist in project folder

pose_landmarker = PoseLandmarker.create_from_model_path(
    model_path=MODEL_PATH,
    running_mode=vision.RunningMode.VIDEO
)

# -----------------------------
# 6. PROCESS VIDEO AND DRAW SKELETON
# -----------------------------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Temporary file to save skeleton overlay video
out_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_temp.name, fourcc, fps, (frame_width, frame_height))

frame_count = 0
landmark_angles = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Mediapipe Pose detection
    result = pose_landmarker.detect_for_video(
        rgb_frame,
        timestamp_ms=int(cap.get(cv2.CAP_PROP_POS_MSEC))
    )

    # Draw skeleton if pose landmarks found
    if result.pose_landmarks:
        for connection in vision.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            h, w, _ = frame.shape
            start = (
                int(result.pose_landmarks[start_idx].x * w),
                int(result.pose_landmarks[start_idx].y * h)
            )
            end = (
                int(result.pose_landmarks[end_idx].x * w),
                int(result.pose_landmarks[end_idx].y * h)
            )
            cv2.line(frame, start, end, (0, 255, 0), 2)
        # Draw points
        for lm in result.pose_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    out.write(frame)
    frame_count += 1

cap.release()
out.release()

# -----------------------------
# 7. DISPLAY VIDEO
# -----------------------------
st.subheader("Skeleton Overlay Video")
video_file = open(out_temp.name, "rb").read()
st.video(video_file)

# -----------------------------
# 8. ANALYZE METRICS
# -----------------------------
# Example: compute simple hip angle metric if available
cap = cv2.VideoCapture(video_path)
peak_angle = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_landmarker.detect_for_video(
        rgb_frame,
        timestamp_ms=int(cap.get(cv2.CAP_PROP_POS_MSEC))
    )
    if result.pose_landmarks:
        # Example: calculate hip angle (simplified)
        h, w, _ = frame.shape
        left_hip = result.pose_landmarks[23]  # left hip
        right_hip = result.pose_landmarks[24]  # right hip
        left_shoulder = result.pose_landmarks[11]
        right_shoulder = result.pose_landmarks[12]
        # approximate peak angle
        angle = abs((left_shoulder.y - left_hip.y) - (right_shoulder.y - right_hip.y)) * 180
        if angle > peak_angle:
            peak_angle = angle
cap.release()

st.subheader("Metrics")
st.metric("Peak Hip Angle", f"{int(peak_angle)}°")

# -----------------------------
# 9. AI FEEDBACK PANEL
# -----------------------------
st.subheader("AI Throw Analysis")

feedback_prompt = f"""
You are a Highland Games coach AI. The athlete just performed a {event}.
The peak hip angle recorded was {int(peak_angle)} degrees.
Please provide detailed feedback on form, technique, and tips for improvement.
"""

if openai.api_key:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": feedback_prompt}]
        )
        feedback = response.choices[0].message.content
        st.markdown(feedback)
    except Exception as e:
        st.error(f"AI feedback unavailable: {e}")
else:
    st.warning("OpenAI API key not set. AI feedback unavailable.")
