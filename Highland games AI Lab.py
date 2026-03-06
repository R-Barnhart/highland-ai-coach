import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import math

# -----------------------------
# STREAMLIT PAGE
# -----------------------------

st.set_page_config(
    page_title="Highland Games AI Coach",
    layout="wide"
)

st.title("🏴 Highland Games AI Throw Coach")

st.write("Upload a throwing video for biomechanical analysis.")

uploaded_video = st.file_uploader(
    "Upload Throw Video",
    type=["mp4", "mov", "avi"]
)

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# ANGLE CALCULATION
# -----------------------------

def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(
        c[1] - b[1], c[0] - b[0]
    ) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle

# -----------------------------
# COACHING FEEDBACK
# -----------------------------

def coaching_feedback(elbow_angle, torque):

    tips = []

    if elbow_angle < 150:
        tips.append("Extend your throwing arm more at release.")

    if elbow_angle > 170:
        tips.append("Good arm extension during release.")

    if torque < 20:
        tips.append("Increase hip-shoulder separation for more power.")

    if torque > 30:
        tips.append("Excellent hip drive and torso rotation.")

    return tips

# -----------------------------
# VIDEO PROCESSING
# -----------------------------

if uploaded_video:

    st.subheader("Video Analysis")

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_video.read())

    cap = cv2.VideoCapture(temp_file.name)

    stframe = st.empty()

    frame_index = 0
    release_frame = None

    elbow_history = []
    torque_history = []

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (960, 540))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb)

        if results.pose_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            landmarks = results.pose_landmarks.landmark

            shoulder = [
                landmarks[12].x,
                landmarks[12].y
            ]

            elbow = [
                landmarks[14].x,
                landmarks[14].y
            ]

            wrist = [
                landmarks[16].x,
                landmarks[16].y
            ]

            hip = [
                landmarks[24].x,
                landmarks[24].y
            ]

            knee = [
                landmarks[26].x,
                landmarks[26].y
            ]

            elbow_angle = calculate_angle(
                shoulder,
                elbow,
                wrist
            )

            hip_angle = calculate_angle(
                shoulder,
                hip,
                knee
            )

            torque = abs(elbow_angle - hip_angle)

            elbow_history.append(elbow_angle)
            torque_history.append(torque)

            if elbow_angle > 165 and release_frame is None:
                release_frame = frame_index

            cv2.putText(
                frame,
                f"Elbow Angle: {int(elbow_angle)}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Hip Torque: {int(torque)}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

        stframe.image(frame, channels="BGR")

        frame_index += 1

    cap.release()

# -----------------------------
# RESULTS
# -----------------------------

    st.success("Analysis Complete")

    if release_frame is not None:
        st.write(f"Estimated Release Frame: {release_frame}")

    if len(elbow_history) > 0:

        avg_elbow = np.mean(elbow_history)
        avg_torque = np.mean(torque_history)

        st.subheader("📊 Throw Metrics")

        col1, col2 = st.columns(2)

        col1.metric(
            "Average Arm Extension",
            int(avg_elbow)
        )

        col2.metric(
            "Average Hip Torque",
            int(avg_torque)
        )

        st.subheader("🏆 Coaching Feedback")

        tips = coaching_feedback(
            avg_elbow,
            avg_torque
        )

        for tip in tips:
            st.write("•", tip)



