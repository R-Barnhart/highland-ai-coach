import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import subprocess

# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE — works on mediapipe==0.10.14 (pinned in requirements.txt)
# ─────────────────────────────────────────────────────────────────────────────
import mediapipe as mp
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Highland Games AI Coach", layout="wide")
st.title("🏴 Highland Games AI Throw Coach")
st.write("Upload a throwing video for biomechanical analysis.")

uploaded_video = st.file_uploader(
    "Upload Throw Video",
    type=["mp4", "mov", "avi", "mpeg4"]
)

# ─────────────────────────────────────────────────────────────────────────────
# DRAWING SPECS  — thick, bright, easy to see
# ─────────────────────────────────────────────────────────────────────────────
LANDMARK_SPEC   = mp_drawing.DrawingSpec(color=(0, 255, 120),  thickness=4, circle_radius=6)
CONNECTION_SPEC = mp_drawing.DrawingSpec(color=(0, 180, 255),  thickness=4)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_angle(a, b, c):
    """Interior angle at joint b using dot product (degrees)."""
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm < 1e-8:
        return 0.0
    cos_a = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def reencode_h264(src: str, dst: str) -> tuple:
    """Re-encode to H.264 mp4 at high quality (CRF 18) for browser playback."""
    for candidate in ["ffmpeg", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0 \
                or os.path.exists(candidate):
            ffmpeg = candidate
            break
    else:
        return False, "ffmpeg not found — add 'ffmpeg' to packages.txt"

    r = subprocess.run(
        [
            ffmpeg, "-y", "-i", src,
            "-vcodec",   "libx264",
            "-preset",   "fast",
            "-crf",      "18",          # 18 = high quality (lower = better)
            "-pix_fmt",  "yuv420p",     # required for Chrome / Safari
            "-movflags", "+faststart",  # stream-ready
            "-an",
            dst,
        ],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) < 500:
        return False, r.stderr[-400:] if r.stderr else "empty output"
    return True, ""


def coaching_feedback(elbow_angle, torque):
    tips = []
    if elbow_angle < 130:
        tips.append("⚠️  Arm significantly bent — extend the throwing arm fully for more distance.")
    elif elbow_angle < 155:
        tips.append("⚠️  Extend your throwing arm more at release for maximum distance.")
    elif elbow_angle > 165:
        tips.append("✅  Good arm extension at release.")
    else:
        tips.append("👍  Acceptable arm extension — aim for full extension (>165°).")

    if torque < 15:
        tips.append("⚠️  Very little hip-shoulder separation. Drive hips first, shoulders follow.")
    elif torque < 25:
        tips.append("⚠️  Increase hip-shoulder separation. Load the hips more in the wind-up.")
    elif torque > 35:
        tips.append("✅  Excellent hip drive and torso rotation — great power generation.")
    else:
        tips.append("👍  Decent hip torque. Work on more aggressive hip initiation.")
    return tips


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_video(input_path: str, raw_out: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None, None, None, None, "cv2 could not open the file."

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # ── Keep native resolution — DO NOT force resize (causes grain) ──────────
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Cap very large videos to 1280 wide to keep processing reasonable
    if native_w > 1280:
        scale   = 1280 / native_w
        out_w   = 1280
        out_h   = int(native_h * scale)
        # make dims even (required by some codecs)
        out_h  += out_h % 2
    else:
        out_w, out_h = native_w, native_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_out, fourcc, fps, (out_w, out_h))

    elbow_history, torque_history = [], []
    release_frame = None
    frame_index   = 0
    landmarks_detected = 0

    bar = st.progress(0, text="Analysing video…")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,           # use heavy model for better accuracy
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.3, # lower threshold = detects more frames
        min_tracking_confidence=0.3,
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize only if needed
            if native_w != out_w:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            # MediaPipe needs RGB
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            # ── Draw wireframe ───────────────────────────────────────────────
            annotated = frame.copy()
            if results.pose_landmarks:
                landmarks_detected += 1
                mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=LANDMARK_SPEC,
                    connection_drawing_spec=CONNECTION_SPEC,
                )

                lm = results.pose_landmarks.landmark
                def pt(i):
                    return [lm[i].x * out_w, lm[i].y * out_h]  # pixel coords

                # ── Check BOTH arms, pick the one more extended ──────────────
                r_elbow = calculate_angle(pt(12), pt(14), pt(16))  # right arm
                l_elbow = calculate_angle(pt(11), pt(13), pt(15))  # left arm
                elbow_angle   = max(r_elbow, l_elbow)
                throwing_side = "R" if r_elbow >= l_elbow else "L"

                # Hip torque — shoulder/hip/knee on throwing side
                if throwing_side == "R":
                    hip_angle = calculate_angle(pt(12), pt(24), pt(26))
                else:
                    hip_angle = calculate_angle(pt(11), pt(23), pt(25))
                torque = abs(elbow_angle - hip_angle)

                elbow_history.append(elbow_angle)
                torque_history.append(torque)

                if elbow_angle > 120 and release_frame is None:
                    release_frame = frame_index

                # ── HUD overlay ──────────────────────────────────────────────
                font      = cv2.FONT_HERSHEY_SIMPLEX
                font_sc   = max(0.6, out_w / 1280)  # scale text to frame size
                thickness = max(2, int(font_sc * 2))
                pad       = int(20 * font_sc)

                # Semi-transparent dark background strip
                overlay = annotated.copy()
                cv2.rectangle(overlay, (0, 0), (int(out_w * 0.42), int(145 * font_sc)), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)

                cv2.putText(annotated, f"Elbow ({throwing_side}): {int(elbow_angle)}°",
                            (pad, int(45 * font_sc)),  font, font_sc, (0, 255, 120),  thickness)
                cv2.putText(annotated, f"Hip Torque: {int(torque)}°",
                            (pad, int(85 * font_sc)),  font, font_sc, (0, 200, 255),  thickness)
                cv2.putText(annotated, f"Frame {frame_index}/{total}",
                            (pad, int(122 * font_sc)), font, font_sc * 0.75, (180, 180, 180), max(1, thickness - 1))

            writer.write(annotated)
            frame_index += 1

            if total > 0:
                bar.progress(min(frame_index / total, 1.0),
                             text=f"Frame {frame_index} / {total}")

    cap.release()
    writer.release()
    bar.empty()

    return elbow_history, torque_history, release_frame, frame_index, landmarks_detected


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if uploaded_video is not None:

    st.subheader("Video Analysis")

    suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_video.read())
        input_path = tmp.name

    raw_path = input_path.replace(suffix, "_raw.mp4")
    web_path = input_path.replace(suffix, "_web.mp4")

    # ── Step 1: pose detection & wireframe ───────────────────────────────────
    elbow_history, torque_history, release_frame, total_frames, lm_count = \
        process_video(input_path, raw_path)

    if elbow_history is None:
        st.error("Could not open video file. Please try a different file.")
        st.stop()

    # Show detection rate so user knows if pose tracking worked
    detect_pct = int(lm_count / max(total_frames, 1) * 100)
    if detect_pct >= 60:
        st.success(f"✅  Analysis complete — {total_frames} frames, pose detected in {detect_pct}% of frames.")
    elif detect_pct > 0:
        st.warning(f"⚠️  Pose only detected in {detect_pct}% of frames. "
                   "Try better lighting or ensure the full body is visible.")
    else:
        st.error("❌  No pose landmarks detected at all. "
                 "Ensure the athlete is clearly visible, well-lit, and the full body is in frame.")

    # ── Step 2: re-encode to H.264 at high quality ───────────────────────────
    with st.spinner("Encoding video for playback…"):
        ok, err = reencode_h264(raw_path, web_path)

    if ok:
        play_path = web_path
    else:
        play_path = raw_path
        st.warning(f"Re-encode failed (`{err}`). "
                   "Make sure `packages.txt` in your repo root contains `ffmpeg` on its own line.")

    with open(play_path, "rb") as f:
        st.video(f.read())

    # ── Step 3: metrics ───────────────────────────────────────────────────────
    if release_frame is not None:
        st.info(f"🎯  Estimated Release Frame: **{release_frame}**")
    else:
        if lm_count > 0:
            st.warning("Release frame not detected — elbow angle stayed below 120°. "
                       "Try filming from the side so the arm extension is clearly visible.")
        # else already shown error above

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


