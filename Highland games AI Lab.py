import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core import RunningMode
import openai

# -----------------------------
# 1. FRONTEND CONFIGURATION
# -----------------------------
st.set_page_config(page_title="Highland Games AI Lab", layout="wide")
st.title("🏴 Highland Games AI Performance Lab")

# Instructions
st.markdown("""
Upload a video of your throw. The app will analyze your throw, overlay a skeleton, 
and provide AI coaching feedback.
""")

# -----------------------------
# 2. OPENAI SETUP
# -----------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")  # Or set your key in env
openai.api_key = OPENAI_API_KEY

# -----------------------------
# 3. EVENT SELECTION
# -----------------------------
event = st.selectbox(
    "Select your event:",
    ["WOB", "WFD", "Hammer", "Stones", "Caber", "Sheaf"]
)

# -----------------------------
# 4. VIDEO UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload Your Throw Video", type=["mp4", "mov"])
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    st.video(video_path)

# -----------------------------
# 5. MEDIAPIPE SETUP (CPU SAFE)
# -----------------------------
MODEL_PATH = "pose_landmarker_lite.task"  # Ensure this file exists in project folder
base_options = BaseOptions(model_asset_path=MODEL_PATH)
pose_options = PoseLandmarker.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.VIDEO
)
pose_landmarker = PoseLandmarker.create_from_options(pose_options)

# -----------------------------
# 6. VIDEO PROCESSING & SKELETON OVERLAY
# -----------------------------
def process_video(video_file):
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose_landmarker.detect_for_video(frame_rgb, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

        if result.pose_landmarks:
            h, w, _ = frame.shape
            for landmark in result.pose_landmarks:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        frame_list.append(frame)

    cap.release()
    return frame_list, fps

processed_frames = []
if uploaded_file:
    processed_frames, fps = process_video(video_path)
    st.success("Pose detection complete!")

# -----------------------------
# 7. AI FEEDBACK PANEL
# -----------------------------
def generate_feedback(event_name, num_frames):
    if not OPENAI_API_KEY:
        return "AI feedback unavailable: API key not set."
    prompt = (
        f"You are a Highland Games coach AI. Analyze a {event_name} throw. "
        f"The throw has {num_frames} video frames analyzed. Give detailed feedback on posture, "
        "technique, and recommendations for improvement."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI feedback unavailable: {str(e)}"

if uploaded_file:
    with st.expander("AI Coaching Feedback"):
        feedback = generate_feedback(event, len(processed_frames))
        st.write(feedback)

# -----------------------------
# 8. DISPLAY SKELETON VIDEO
# -----------------------------
if processed_frames:
    st.subheader("Skeleton Overlay Video Preview")
    video_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w, _ = processed_frames[0].shape
    out = cv2.VideoWriter(video_temp_file.name, fourcc, fps, (w, h))

    for f in processed_frames:
        out.write(f)
    out.release()

    video_bytes = open(video_temp_file.name, 'rb').read()
    st.video(video_bytes)
)

