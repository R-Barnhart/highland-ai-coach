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
    """Return the interior angle at point b (in degrees)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def reencode_for_browser(src: str, dst: str) -> tuple:
    """
    Re-encode to H.264/mp4 so every browser can play it.
    Returns (success: bool, error_message: str).
    """
    ffmpeg_path = None
    for candidate in ["ffmpeg", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        result = subprocess.run(["which", candidate.split("/")[-1]],
                                capture_output=True, text=True)
        if result.returncode == 0 or os.path.exists(candidate):
            ffmpeg_path = candidate
            break

    if ffmpeg_path is None:
        return False, "ffmpeg not found. Add 'ffmpeg' to packages.txt in your repo root."

    result = subprocess.run(
        [
            ffmpeg_path, "-y",
            "-i", src,
            "-vcodec", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",    # required for broad browser support
            "-movflags", "+faststart", # moov atom at front for streaming
            "-an",                     # drop audio
            dst,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        return False, result.stderr[-600:] if result.stderr else "Unknown ffmpeg error"

    if not os.path.exists(dst) or os.path.getsize(dst) < 100:
        return False, "ffmpeg produced an empty output file"

    return True, ""


def coaching_feedback(elbow_angle, torque):
    tips = []
    if elbow_angle < 130:
        tips.append("⚠️  Arm is significantly bent at release — extend fully for more distance.")
    elif elbow_angle < 155:
        tips.append("⚠️  Extend your throwing arm more at release for maximum distance.")
    elif elbow_angle > 165:
        tips.append("✅  Good arm extension at release.")
    else:
        tips.append("👍  Arm extension is acceptable — aim for full extension (>165°).")

    if torque < 15:
        tips.append("⚠️  Very little hip-shoulder separation detected. Drive hips first, shoulders follow.")
    elif torque < 25:
        tips.append("⚠️  Increase hip-shoulder separation. Load the hips more aggressively in the wind-up.")
    elif torque > 35:
        tips.append("✅  Excellent hip drive and torso rotation — great power generation.")
    else:
        tips.append("👍  Decent hip torque. Keep working on earlier hip initiation.")
    return tips

# -----------------------------
# VIDEO PROCESSING
# -----------------------------

def process_video(input_path, raw_output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None, None, None, None, "Could not open video file."

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w, out_h = 960, 540

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
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
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

                # ── Check BOTH arms, use whichever is more extended ──────────
                # Right arm: shoulder=12, elbow=14, wrist=16, hip=24, knee=26
                # Left arm:  shoulder=11, elbow=13, wrist=15, hip=23, knee=25
                r_elbow = calculate_angle(pt(12), pt(14), pt(16))
                l_elbow = calculate_angle(pt(11), pt(13), pt(15))
                elbow_angle = max(r_elbow, l_elbow)
                throwing_side = "R" if r_elbow >= l_elbow else "L"

                # Hip torque — use same side as throwing arm
                if throwing_side == "R":
                    torque = abs(elbow_angle - calculate_angle(pt(12), pt(24), pt(26)))
                else:
                    torque = abs(elbow_angle - calculate_angle(pt(11), pt(23), pt(25)))

                elbow_history.append(elbow_angle)
                torque_history.append(torque)

                # Release = first frame where best elbow angle exceeds 130°
                if elbow_angle > 130 and release_frame is None:
                    release_frame = frame_index

                cv2.putText(annotated, f"Elbow ({throwing_side}): {int(elbow_angle)} deg",
                            (20, 45),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),   2)
                cv2.putText(annotated, f"Torque: {int(torque)} deg",
                            (20, 88),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 2)
                cv2.putText(annotated, f"Frame {frame_index}/{total}",
                            (20, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)

            writer.write(annotated)
            frame_index += 1

            if total > 0:
                bar.progress(min(frame_index / total, 1.0),
                             text=f"Processing frame {frame_index} / {total}...")

    cap.release()
    writer.release()
    bar.empty()
    return elbow_history, torque_history, release_frame, frame_index, None

# -----------------------------
# MAIN
# -----------------------------

if uploaded_video is not None:

    st.subheader("Video Analysis")

    suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in.write(uploaded_video.read())
        input_path = tmp_in.name

    raw_path = input_path.replace(suffix, "_raw.mp4")
    web_path = input_path.replace(suffix, "_web.mp4")

    # ── Step 1: pose detection ────────────────────────────────────────────────
    elbow_history, torque_history, release_frame, total_frames, vid_err = \
        process_video(input_path, raw_path)

    if elbow_history is None:
        st.error(f"Could not open video: {vid_err}")
        st.stop()

    st.success(f"✅  Analysis complete — {total_frames} frames processed.")

    # ── Step 2: re-encode to H.264 for browser playback ──────────────────────
    with st.spinner("Preparing video for playback..."):
        encoded_ok, enc_error = reencode_for_browser(raw_path, web_path)

    if encoded_ok:
        play_path = web_path
    else:
        play_path = raw_path
        st.warning(
            f"⚠️  Video re-encode failed: `{enc_error}`\n\n"
            "**Fix:** make sure your repo contains a `packages.txt` file with `ffmpeg` on its own line."
        )

    with open(play_path, "rb") as vf:
        st.video(vf.read())

    # ── Step 3: metrics ───────────────────────────────────────────────────────
    if release_frame is not None:
        st.info(f"🎯  Estimated Release Frame: **{release_frame}**")
    else:
        st.warning(
            "Release frame not detected. Ensure the athlete's full arms are clearly "
            "visible in the video and the pose skeleton appeared on screen."
        )

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

    for p in [input_path, raw_path, web_path]:
        try:
            os.unlink(p)
        except Exception:
            pass

