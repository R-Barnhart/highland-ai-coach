import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker
from mediapipe.tasks.python.vision.core import BaseOptions, RunningMode
import openai

# -----------------------------
# FRONTEND CONFIGURATION
# -----------------------------
st.set_page_config(page_title="Highland Games AI Lab", layout="wide")
st.title("Highland Games AI Performance Lab")
st.write(
    """
Upload your throw video, select the event, and get AI feedback along with skeleton overlay.
"""
)

# -----------------------------
# USER INPUTS
# -----------------------------
event = st.selectbox(
    "Select Event",
    ["WOB", "WFD", "Hammer", "Stones", "Caber", "Sheaf"]
)

uploaded_video = st.file_uploader("Upload Your Throw Video", type=["mp4", "mov"])

# -----------------------------
# ENVIRONMENT FIXES (CPU ONLY)
# -----------------------------
# Disable GPU to prevent EGL/GL errors
os.environ["MEDIAPIPE_DISABLE_GPU"] = "true"

# -----------------------------
# MEDIAPIPE TASKS SETUP
# -----------------------------
MODEL_PATH = "pose_landmarker_lite.task"  # Make sure the model file exists

base_options = BaseOptions(model_asset_path=MODEL_PATH)
pose_options = PoseLandmarker.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.VIDEO
)

pose_landmarker = PoseLandmarker.create_from_options(pose_options)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def draw_skeleton(frame, landmarks):
    h, w, _ = frame.shape
    # Simple skeleton connections (pairs of indices)
    connections = [
        (0, 1), (1, 2), (2, 3),  # Example connections, adapt to full model
        (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9)
    ]
    for start_idx, end_idx in connections:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            x1, y1 = int(landmarks[start_idx][0] * w), int(landmarks[start_idx][1] * h)
            x2, y2 = int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for lm in landmarks:
        cx, cy = int(lm[0] * w), int(lm[1] * h)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    return frame

def get_ai_feedback(metrics_dict, event):
    prompt = f"""
Analyze the following throw data for {event}:
{metrics_dict}

Provide concise, actionable feedback to improve performance.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception:
        return "AI feedback unavailable."

# -----------------------------
# VIDEO PROCESSING
# -----------------------------
if uploaded_video:
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)

    # Prepare video writer for overlay output
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(output_file.name, fourcc, fps, (width, height))

    peak_angle = 0
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose_landmarker.detect_for_video(frame_rgb, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

        if result.pose_landmarks:
            landmarks = [(lm.x, lm.y) for lm in result.pose_landmarks.landmark]
            frame = draw_skeleton(frame, landmarks)
            # Example metric calculation: track peak hip angle (simplified)
            hip_angle = 0
            if len(landmarks) > 12:
                hip_angle = np.degrees(np.arctan2(
                    landmarks[12][1]-landmarks[24][1],
                    landmarks[12][0]-landmarks[24][0]
                ))
            if hip_angle > peak_angle:
                peak_angle = hip_angle

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # -----------------------------
    # STREAMLIT VIDEO DISPLAY
    # -----------------------------
    st.subheader("Skeleton Overlay Video")
    video_file = open(output_file.name, 'rb').read()
    st.video(video_file)

    # -----------------------------
    # COACHING PANEL
    # -----------------------------
    st.subheader("AI Coaching Feedback")
    metrics = {"peak_hip_angle": int(peak_angle)}
    feedback = get_ai_feedback(metrics, event)
    st.text(feedback)
