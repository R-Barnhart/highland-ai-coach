import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os

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

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# -----------------------------
# ANGLE CALCULATION
# -----------------------------

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) \
            - np.arctan2(a[1] - b[1], a[0] - b[0])

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
        tips.append("Extend your throwing arm more at release for maximum distance.")
    elif elbow_angle > 165:
        tips.append("Good arm extension at release.")
    else:
        tips.append("Arm extension is acceptable — work toward full extension (>165 degrees).")

    if torque < 20:
        tips.append("Increase hip-shoulder separation. Drive your hips before your shoulders rotate.")
    elif torque > 30:
        tips.append("Excellent hip drive and torso rotation — good power generation.")
    else:
        tips.append("Decent hip torque. Focus on loading the hips more in the wind-up phase.")

    return tips

# -----------------------------
# VIDEO PROCESSING
# -----------------------------

def process_video(input_path, output_path):
    """
    Run MediaPipe Pose on every frame, draw skeleton wireframe,
    write annotated video to output_path, and collect metrics.
    Returns (elbow_history, torque_history, release_frame, total_frames).
    """

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        return None, None, None, None

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    out_w, out_h = 960, 540
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    elbow_history  = []
    torque_history = []
    release_frame  = None
    frame_index    = 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bar   = st.progress(0, text="Processing video...")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame     = cv2.resize(frame, (out_w, out_h))
            rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = pose.process(rgb)
            annotated = frame.copy()

            if results.pose_landmarks:

                mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 180), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 80,  80), thickness=2),
                )

                lm = results.pose_landmarks.landmark

                def pt(idx):
                    return [lm[idx].x, lm[idx].y]

                # Right-side landmarks
                shoulder = pt(12)
                elbow    = pt(14)
                wrist    = pt(16)
                hip      = pt(24)
                knee     = pt(26)

                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                hip_angle   = calculate_angle(shoulder, hip, knee)
                torque      = abs(elbow_angle - hip_angle)

                elbow_history.append(elbow_angle)
                torque_history.append(torque)

                if elbow_angle > 165 and release_frame is None:
                    release_frame = frame_index

                cv2.putText(annotated, f"Elbow Angle: {int(elbow_angle)}",
                            (30, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated, f"Hip Torque:  {int(torque)}",
                            (30, 80),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
                cv2.putText(annotated, f"Frame: {frame_index}",
                            (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            writer.write(annotated)
            frame_index += 1

            if total > 0:
                bar.progress(
                    min(frame_index / total, 1.0),
                    text=f"Processing frame {frame_index}/{total}..."
                )

    cap.release()
    writer.release()
    bar.empty()

    return elbow_history, torque_history, release_frame, frame_index

# -----------------------------
# MAIN APP LOGIC
# -----------------------------

if uploaded_video is not None:

    st.subheader("Video Analysis")

    # FIX: Write temp file and CLOSE it before OpenCV reads it
    suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in.write(uploaded_video.read())
        input_path = tmp_in.name
    # File is now flushed and closed — safe for OpenCV

    output_path = input_path.replace(suffix, "_annotated.mp4")

    elbow_history, torque_history, release_frame, total_frames = \
        process_video(input_path, output_path)

    if elbow_history is None:
        st.error("Could not open the video file. Please try a different file.")
        st.stop()

    # -----------------------------
    # DISPLAY ANNOTATED VIDEO
    # -----------------------------

    st.success(f"Analysis complete — {total_frames} frames processed.")

    # FIX: Display finished video file instead of slow frame-by-frame streaming
    if os.path.exists(output_path):
        with open(output_path, "rb") as vf:
            st.video(vf.read())
    else:
        st.warning("Annotated video could not be saved.")

    # -----------------------------
    # RESULTS
    # -----------------------------

    if release_frame is not None:
        st.write(f"Estimated Release Frame: {release_frame}")
    else:
        st.warning("Release frame not detected (elbow never exceeded 165 degrees).")

    if len(elbow_history) > 0:

        avg_elbow  = float(np.mean(elbow_history))
        avg_torque = float(np.mean(torque_history))
        max_elbow  = float(np.max(elbow_history))
        max_torque = float(np.max(torque_history))

        st.subheader("Throw Metrics")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Arm Extension", f"{int(avg_elbow)} deg")
        col2.metric("Max Arm Extension", f"{int(max_elbow)} deg")
        col3.metric("Avg Hip Torque",    f"{int(avg_torque)} deg")
        col4.metric("Max Hip Torque",    f"{int(max_torque)} deg")

        st.subheader("Angle History")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.caption("Elbow Angle per Frame")
            st.line_chart(elbow_history)
        with chart_col2:
            st.caption("Hip Torque per Frame")
            st.line_chart(torque_history)

        st.subheader("Coaching Feedback")
        tips = coaching_feedback(avg_elbow, avg_torque)
        for tip in tips:
            st.write("-", tip)

    # Clean up temp files
    try:
        os.unlink(input_path)
        os.unlink(output_path)
    except Exception:
        pass



