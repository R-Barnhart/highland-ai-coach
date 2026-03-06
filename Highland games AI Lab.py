import streamlit as st
import cv2
import numpy as np
import tempfile
import mediapipe as mp
import openai

from mediapipe.tasks.python.vision import PoseLandmarker
from mediapipe.tasks.python.vision import PoseLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions


# -----------------------
# PAGE SETUP
# -----------------------

st.set_page_config(page_title="Highland Games AI Lab", layout="wide")
st.title("🏴 Highland Games AI Lab")

# -----------------------
# SIDEBAR
# -----------------------

event = st.sidebar.selectbox(
    "Select Event",
    ["WOB", "WFD", "Hammer", "Stones", "Caber", "Sheaf"]
)

api_key = st.sidebar.text_input("OpenAI API Key", type="password")
openai.api_key = api_key

uploaded_file = st.file_uploader("Upload Throw Video", type=["mp4","mov","avi"])


# -----------------------
# LOAD POSE MODEL
# -----------------------

MODEL_PATH = "pose_landmarker_lite.task"

@st.cache_resource
def load_pose_model():

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO
    )

    return PoseLandmarker.create_from_options(options)

pose_model = load_pose_model()


# -----------------------
# SKELETON DRAW FUNCTION
# -----------------------

def draw_skeleton(frame, landmarks):

    h, w, _ = frame.shape
    pts = []

    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        pts.append((x,y))

    # simplified connection map
    connections = [
        (11,13),(13,15),
        (12,14),(14,16),
        (11,12),
        (23,24),
        (11,23),(12,24),
        (23,25),(25,27),
        (24,26),(26,28)
    ]

    for a,b in connections:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (0,255,0),2)

    for p in pts:
        cv2.circle(frame,p,4,(0,0,255),-1)

    return frame


# -----------------------
# VIDEO PROCESSING
# -----------------------

if uploaded_file:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    frame_count = 0

    hip_drive_vals=[]
    rotation_vals=[]
    balance_vals=[]

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # Skip frames for speed
        if frame_count % 3 != 0:
            frame_count +=1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp = frame_count * 33

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = pose_model.detect_for_video(
            mp_image,
            timestamp
        )

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]

            frame = draw_skeleton(frame, landmarks)

            hip = landmarks[24]
            shoulder = landmarks[12]
            ankle = landmarks[28]

            hip_drive = abs(hip.x - shoulder.x)
            rotation = abs(shoulder.z)
            balance = abs(hip.x - ankle.x)

            hip_drive_vals.append(hip_drive)
            rotation_vals.append(rotation)
            balance_vals.append(balance)

        stframe.image(frame, channels="BGR")

        frame_count += 1

    cap.release()

    # -----------------------
    # METRICS
    # -----------------------

    if hip_drive_vals:

        avg_hip = np.mean(hip_drive_vals)
        avg_rot = np.mean(rotation_vals)
        avg_bal = np.mean(balance_vals)

        st.subheader("Biomechanics Metrics")

        col1,col2,col3 = st.columns(3)

        col1.metric("Hip Drive", round(avg_hip,3))
        col2.metric("Rotation", round(avg_rot,3))
        col3.metric("Balance", round(avg_bal,3))


    # -----------------------
    # AI COACH
    # -----------------------

    if api_key and st.button("Generate AI Coaching"):

        prompt = f"""
        Athlete performed a {event} throw.

        Metrics:
        Hip Drive: {avg_hip}
        Rotation: {avg_rot}
        Balance: {avg_bal}

        Provide coaching feedback and drills.
        """

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}]
        )

        feedback = response["choices"][0]["message"]["content"]

        st.subheader("AI Coach Feedback")
        st.write(feedback)





