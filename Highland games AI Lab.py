import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import subprocess

# -----------------------------
# MEDIAPIPE — safe import
# -----------------------------
try:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except Exception:
    import mediapipe as mp
    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Highland Games AI Coach", layout="wide")
st.title("🏴 Highland Games AI Throw Coach")
st.write("Upload a throwing video for biomechanical analysis.")

uploaded_video = st.file_uploader(
    "Upload Throw Video",
    type=["mp4", "mov", "avi", "mpeg4"]
)

# -----------------------------
# HELPERS
# -----------------------------

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = (
        np.arctan2(c[1] - b[1], c[0] - b[0])
        - np.arctan2(a[1] - b[1], a[0] - b[0])
    )
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


def reencode_for_browser(src: str, dst: str) -> bool:
    """Re-encode src to H.264/AAC mp4 so every browser can play it."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", src,
                "-vcodec", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",   # required for broad browser support
                "-movflags", "+faststart", # puts metadata at front for streaming
                "-an",                    # no audio needed
                dst,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
        )
        return result.returncode == 0 and os.path.exists(dst)
    except Exception:
        return False


def coaching_feedback(elbow_angle, torque):
    tips = []
    if elbow_angle < 150:
        tips.append("⚠️  Extend your throwing arm more at release for maximum distance.")
    elif elbow_angle > 165:
        tips.append("✅  Good arm extension at release.")
    else:
        tips.append("👍  Arm extension is acceptable — aim for full extension (>165°).")

    if torque < 20:
        tips.append("⚠️  Increase hip-shoulder separation. Drive hips before shoulders rotate.")
    elif torque > 30:
        tips.append("✅  Excellent hip drive and torso rotation — great power generation.")
    else:
        tips.append("👍  Decent hip torque. Load the hips more aggressively in the wind-up.")
    return tips

# -----------------------------
# VIDEO PROCESSING
# -----------------------------

def process_video(input_path, raw_output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None, None, None, None

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w, out_h = 960, 540

    # OpenCV writes mp4v first; we'll re-encode to H.264 afterward
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (out_w, out_h))

    elbow_history, torque_history = [], []
    release_frame = None
    frame_index   = 0
    bar = st.progress(0, text="Processing video...")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    ) as pose:

        while True:
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
                def pt(i): return [lm[i].x, lm[i].y]

                # Right-side landmarks
                shoulder = pt(12); elbow = pt(14); wrist = pt(16)
                hip      = pt(24); knee  = pt(26)

                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                hip_angle   = calculate_angle(shoulder, hip, knee)
                torque      = abs(elbow_angle - hip_angle)

                elbow_history.append(elbow_angle)
                torque_history.append(torque)

                # Lowered threshold: 140° catches partial extension too
                if elbow_angle > 140 and release_frame is None:
                    release_frame = frame_index

                cv2.putText(annotated, f"Elbow: {int(elbow_angle)} deg",
                            (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),   2)
                cv2.putText(annotated, f"Torque: {int(torque)} deg",
                            (20, 80),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 2)
                cv2.putText(annotated, f"Frame: {frame_index}/{total}",
                            (20, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)

            writer.write(annotated)
            frame_index += 1

            if total > 0:
                bar.progress(
                    min(frame_index / total, 1.0),
                    text=f"Processing frame {frame_index} / {total}..."
                )

    cap.release()
    writer.release()
    bar.empty()
    return elbow_history, torque_history, release_frame, frame_index

# -----------------------------
# MAIN
# -----------------------------

if uploaded_video is not None:

    st.subheader("Video Analysis")

    suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"

    # Write upload to disk and close before OpenCV reads it
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in.write(uploaded_video.read())
        input_path = tmp_in.name

    raw_path  = input_path.replace(suffix, "_raw.mp4")       # mp4v output from OpenCV
    web_path  = input_path.replace(suffix, "_web.mp4")       # H.264 re-encode for browser

    # ── Step 1: pose detection ────────────────────────────────────────────────
    elbow_history, torque_history, release_frame, total_frames = \
        process_video(input_path, raw_path)

    if elbow_history is None:
        st.error("Could not open the video file. Please try a different file.")
        st.stop()

    st.success(f"✅  Analysis complete — {total_frames} frames processed.")

    # ── Step 2: re-encode to H.264 so browser can play it ────────────────────
    with st.spinner("Preparing video for playback..."):
        encoded_ok = reencode_for_browser(raw_path, web_path)

    play_path = web_path if encoded_ok else raw_path

    with open(play_path, "rb") as vf:
        video_bytes = vf.read()

    st.video(video_bytes)

    if not encoded_ok:
        st.caption("⚠️  ffmpeg re-encode failed — showing raw OpenCV output (may not play in all browsers).")

    # ── Step 3: metrics ───────────────────────────────────────────────────────
    if release_frame is not None:
        st.info(f"🎯  Estimated Release Frame: **{release_frame}**")
    else:
        st.warning("Release frame not detected. Check that the full arm is visible in the video.")

    if elbow_history:
        avg_elbow  = float(np.mean(elbow_history))
        avg_torque = float(np.mean(torque_history))
        max_elbow  = float(np.max(elbow_history))
        max_torque = float(np.max(torque_history))

        st.subheader("📊 Throw Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Arm Extension", f"{int(avg_elbow)}°")
        c2.metric("Max Arm Extension", f"{int(max_elbow)}°")
        c3.metric("Avg Hip Torque",    f"{int(avg_torque)}°")
        c4.metric("Max Hip Torque",    f"{int(max_torque)}°")

        st.subheader("📈 Angle History")
        ch1, ch2 = st.columns(2)
        with ch1:
            st.caption("Elbow Angle per Frame")
            st.line_chart(elbow_history)
        with ch2:
            st.caption("Hip Torque per Frame")
            st.line_chart(torque_history)

        st.subheader("🏆 Coaching Feedback")
        for tip in coaching_feedback(avg_elbow, avg_torque):
            st.write(tip)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    for p in [input_path, raw_path, web_path]:
        try:
            os.unlink(p)
        except Exception:
            pass


