import streamlit as st
import cv2
import numpy as np
import tempfile
import sqlite3
import mediapipe as mp
import pandas as pd
from datetime import datetime
import openai

# -----------------------
# PAGE SETUP
# -----------------------

st.set_page_config(
    page_title="Highland Games AI Coach",
    layout="wide"
)

st.title("🏴 Highland Games AI Coach")

# -----------------------
# DATABASE
# -----------------------

conn = sqlite3.connect("throws.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS throws(
id INTEGER PRIMARY KEY AUTOINCREMENT,
date TEXT,
hip_drive REAL,
rotation REAL,
balance REAL
)
""")

conn.commit()

# -----------------------
# MEDIAPIPE SETUP
# -----------------------

from mediapipe.tasks.python.vision import PoseLandmarker
from mediapipe.tasks.python.vision import PoseLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_PATH = "pose_landmarker_lite.task"

@st.cache_resource
def load_model():

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO
    )

    return PoseLandmarker.create_from_options(options)

pose_model = load_model()

# Skeleton connections
mp_pose = mp.solutions.pose
connections = mp_pose.POSE_CONNECTIONS

# -----------------------
# VIDEO UPLOAD
# -----------------------

uploaded_file = st.file_uploader("Upload Throw Video")

# -----------------------
# PROCESS VIDEO
# -----------------------

if uploaded_file:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    metrics = {
        "hip_drive": [],
        "rotation": [],
        "balance": []
    }

    frame_count = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # Skip frames for speed
        if frame_count % 3 != 0:
            frame_count += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # FIXED timestamp
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

            # Extract important joints
            hip = landmarks[24]
            shoulder = landmarks[12]
            ankle = landmarks[28]

            hip_drive = abs(hip.x - shoulder.x)
            rotation = abs(shoulder.z)
            balance = abs(hip.x - ankle.x)

            metrics["hip_drive"].append(hip_drive)
            metrics["rotation"].append(rotation)
            metrics["balance"].append(balance)

            # -----------------------
            # DRAW SKELETON
            # -----------------------

            h, w, _ = frame.shape

            points = []

            for lm in landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))

            for connection in connections:

                start = connection[0]
                end = connection[1]

                if start < len(points) and end < len(points):

                    cv2.line(
                        frame,
                        points[start],
                        points[end],
                        (0,255,0),
                        2
                    )

            for point in points:

                cv2.circle(
                    frame,
                    point,
                    4,
                    (0,0,255),
                    -1
                )

        stframe.image(frame, channels="BGR")

        frame_count += 1

    cap.release()

# -----------------------
# METRICS
# -----------------------

    avg_hip = np.mean(metrics["hip_drive"])
    avg_rotation = np.mean(metrics["rotation"])
    avg_balance = np.mean(metrics["balance"])

    st.subheader("Biomechanics Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Hip Drive", round(avg_hip,3))
    col2.metric("Rotation", round(avg_rotation,3))
    col3.metric("Balance", round(avg_balance,3))

# -----------------------
# SAVE THROW
# -----------------------

    if st.button("Save Throw"):

        c.execute("""
        INSERT INTO throws(date, hip_drive, rotation, balance)
        VALUES(?,?,?,?)
        """,(
            datetime.now().strftime("%Y-%m-%d"),
            float(avg_hip),
            float(avg_rotation),
            float(avg_balance)
        ))

        conn.commit()

        st.success("Throw saved!")

# -----------------------
# AI COACH
# -----------------------

    if st.button("Generate AI Coaching"):

        prompt = f"""
        Highland Games athlete metrics:

        Hip Drive: {avg_hip}
        Rotation: {avg_rotation}
        Balance: {avg_balance}

        Provide coaching feedback and drills.
        """

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}]
        )

        feedback = response["choices"][0]["message"]["content"]

        st.subheader("AI Coach Feedback")

        st.write(feedback)

# -----------------------
# HISTORY DASHBOARD
# -----------------------

st.header("Throw History")

df = pd.read_sql_query(
    "SELECT * FROM throws",
    conn
)

if len(df) > 0:

    st.line_chart(df["hip_drive"])
    st.line_chart(df["rotation"])
    st.line_chart(df["balance"])

    st.dataframe(df)

else:

    st.info("No throws recorded yet.")


