import streamlit as st
import cv2
import tempfile
import os
import openai
import numpy as np

from mediapipe.tasks.python.vision import PoseLandmarker
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="Highland Games AI Lab",
    layout="wide"
)

st.title("🏴 Highland Games AI Performance Lab")

# -----------------------
# EVENT SELECTOR
# -----------------------
event = st.selectbox(
    "Select Event",
    ["WOB", "WFD", "Hammer", "Stones", "Caber", "Sheaf"]
)

# -----------------------
# OPENAI SETUP
# -----------------------
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]

# -----------------------
# LOAD MEDIAPIPE MODEL (ONLY ONCE)
# -----------------------
@st.cache_resource
def load_pose_model():

    MODEL_PATH = "pose_landmarker_lite.task"

    if not os.path.exists(MODEL_PATH):
        st.error("Pose model file missing. Add pose_landmarker_lite.task to repo.")
        st.stop()

    options = PoseLandmarker.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO
    )

    return PoseLandmarker.create_from_options(options)

pose_landmarker = load_pose_model()

# -----------------------
# VIDEO UPLOAD
# -----------------------
uploaded_file = st.file_uploader(
    "Upload your throw video",
    type=["mp4", "mov"]
)

if uploaded_file is None:
    st.info("Upload a video to begin analysis.")
    st.stop()

# Save temp video
temp_video = tempfile.NamedTemporaryFile(delete=False)
temp_video.write(uploaded_file.read())
video_path = temp_video.name

st.subheader("Original Video")
st.video(video_path)

# -----------------------
# PROCESS VIDEO
# -----------------------
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video
out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(out_file.name, fourcc, fps, (width, height))

frame_count = 0
hip_heights = []

while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

    result = pose_landmarker.detect_for_video(rgb, timestamp)

    if result.pose_landmarks:

        landmarks = result.pose_landmarks

        # Draw joints
        for lm in landmarks:

            x = int(lm.x * width)
            y = int(lm.y * height)

            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        # Track hip height (simple metric)
        left_hip = landmarks[23]
        hip_heights.append(left_hip.y)

    out.write(frame)
    frame_count += 1

cap.release()
out.release()

# -----------------------
# SHOW SKELETON VIDEO
# -----------------------
st.subheader("Skeleton Overlay")
st.video(out_file.name)

# -----------------------
# METRICS
# -----------------------
st.subheader("Performance Metrics")

if hip_heights:

    peak_drive = min(hip_heights)

    st.metric(
        "Peak Hip Drive",
        f"{round(peak_drive,3)}"
    )

# -----------------------
# AI COACHING
# -----------------------
st.subheader("AI Coaching Feedback")

if "OPENAI_API_KEY" not in st.secrets:

    st.warning("Add OPENAI_API_KEY to Streamlit secrets to enable AI coaching.")

else:

    prompt = f"""
You are a professional Highland Games throwing coach.

The athlete performed a {event} throw.

Peak hip drive metric: {round(peak_drive,3)}

Give clear coaching advice:
- technique issues
- body positioning
- power generation
- 2 specific drills to improve
"""

    try:

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role":"user","content":prompt}],
            max_tokens=300
        )

        feedback = response.choices[0].message.content

        st.write(feedback)

    except Exception as e:

        st.error(f"AI feedback failed: {e}")
