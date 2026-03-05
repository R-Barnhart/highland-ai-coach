import streamlit as st
import cv2
import numpy as np
import sqlite3
import tempfile
import os
import math
import pandas as pd
import openai

from mediapipe.tasks.python.vision import PoseLandmarker
from mediapipe.tasks.python.vision import PoseLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Highland Games AI Coach", layout="wide")

st.title("🏴 Highland Games AI Performance Lab")

# ----------------------------
# DATABASE
# ----------------------------
conn = sqlite3.connect("throws.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS throws(
id INTEGER PRIMARY KEY AUTOINCREMENT,
event TEXT,
date TEXT,
hip_drive REAL,
release_angle REAL,
rotation_speed REAL,
separation REAL
)
""")

conn.commit()

# ----------------------------
# OPENAI KEY
# ----------------------------
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]

# ----------------------------
# LOAD POSE MODEL
# ----------------------------
@st.cache_resource
def load_model():

    MODEL_PATH = "pose_landmarker_lite.task"

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO
    )

    return PoseLandmarker.create_from_options(options)

pose_model = load_model()

# ----------------------------
# EVENT SELECTOR
# ----------------------------
event = st.selectbox(
    "Select Event",
    ["WOB","WFD","Hammer","Stones","Caber","Sheaf"]
)

# ----------------------------
# VIDEO UPLOAD
# ----------------------------
uploaded = st.file_uploader("Upload Throw Video", type=["mp4","mov"])

if uploaded:

    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(uploaded.read())
    video_path = temp.name

    st.subheader("Original Video")
    st.video(video_path)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_file = tempfile.NamedTemporaryFile(delete=False,suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_file.name,fourcc,fps,(width,height))

    hip_positions=[]
    shoulder_angles=[]
    wrist_speed=[]
    release_angles=[]

    prev_wrist=None
    frame_count=0

    while True:

        ret,frame = cap.read()

        if not ret:
            break

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        timestamp=int(cap.get(cv2.CAP_PROP_POS_MSEC))

        result=pose_model.detect_for_video(rgb,timestamp)

        if result.pose_landmarks:

            lm=result.pose_landmarks

            for p in lm:
                x=int(p.x*width)
                y=int(p.y*height)
                cv2.circle(frame,(x,y),4,(0,255,0),-1)

            # HIP DRIVE
            hip=lm[23]
            hip_positions.append(hip.y)

            # SHOULDER ROTATION
            ls=lm[11]
            rs=lm[12]

            angle=math.atan2(
                rs.y-ls.y,
                rs.x-ls.x
            )

            shoulder_angles.append(angle)

            # WRIST SPEED
            wrist=lm[16]

            if prev_wrist:
                dx=wrist.x-prev_wrist.x
                dy=wrist.y-prev_wrist.y
                speed=math.sqrt(dx*dx+dy*dy)
                wrist_speed.append(speed)

            prev_wrist=wrist

            # RELEASE ANGLE
            elbow=lm[14]

            rel_angle=math.degrees(
                math.atan2(
                    wrist.y-elbow.y,
                    wrist.x-elbow.x
                )
            )

            release_angles.append(rel_angle)

        out.write(frame)
        frame_count+=1

    cap.release()
    out.release()

    st.subheader("Pose Tracking")
    st.video(out_file.name)

    # ----------------------------
    # METRICS
    # ----------------------------

    peak_hip=min(hip_positions) if hip_positions else 0
    release=np.mean(release_angles) if release_angles else 0
    separation=np.std(shoulder_angles) if shoulder_angles else 0

    rotation_speed=0

    if len(shoulder_angles)>5:
        diffs=np.diff(shoulder_angles)
        rotation_speed=np.mean(np.abs(diffs))*fps

    col1,col2,col3,col4=st.columns(4)

    col1.metric("Hip Drive",round(peak_hip,3))
    col2.metric("Release Angle",round(release,2))
    col3.metric("Hip-Shoulder Separation",round(separation,3))
    col4.metric("Rotation Speed",round(rotation_speed,3))

    # ----------------------------
    # THROW PHASE DETECTION
    # ----------------------------

    phase="Unknown"

    if wrist_speed:

        peak_speed=max(wrist_speed)

        if peak_speed<0.01:
            phase="Wind-up"

        elif peak_speed<0.03:
            phase="Rotation"

        elif peak_speed<0.06:
            phase="Power Phase"

        else:
            phase="Release"

    st.subheader("Detected Throw Phase")
    st.write(phase)

    # ----------------------------
    # SAVE THROW
    # ----------------------------

    cursor.execute("""
    INSERT INTO throws(event,date,hip_drive,release_angle,rotation_speed,separation)
    VALUES(datetime('now'),?,?,?,?,?)
    """,(event,peak_hip,release,rotation_speed,separation))

    conn.commit()

# ----------------------------
# ATHLETE DASHBOARD
# ----------------------------

st.header("📊 Throw History")

df=pd.read_sql_query("SELECT * FROM throws",conn)

if not df.empty:

    st.dataframe(df)

    st.subheader("Progress Charts")

    st.line_chart(df["hip_drive"])
    st.line_chart(df["release_angle"])
    st.line_chart(df["rotation_speed"])

# ----------------------------
# AI COACH
# ----------------------------

st.header("🤖 AI Coaching")

if not df.empty and "OPENAI_API_KEY" in st.secrets:

    latest=df.iloc[-1]

    prompt=f"""
You are a Highland Games throwing coach.

Event: {latest.event}

Metrics:
Hip Drive: {latest.hip_drive}
Release Angle: {latest.release_angle}
Rotation Speed: {latest.rotation_speed}
Hip Shoulder Separation: {latest.separation}

Provide:
• Technique critique
• Power generation advice
• 3 training drills
"""

    try:

        response=openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role":"user","content":prompt}],
            max_tokens=300
        )

        st.write(response.choices[0].message.content)

    except Exception as e:

        st.error(e)

