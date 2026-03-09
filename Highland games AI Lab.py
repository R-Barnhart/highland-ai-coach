import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import subprocess

# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE  (requires mediapipe==0.10.14 in requirements.txt)
# model_complexity=1 uses the bundled lite model — no download needed.
# model_complexity=2 tries to download heavy.tflite at runtime which
# fails on Streamlit Cloud due to venv write permissions.
# ─────────────────────────────────────────────────────────────────────────────
import mediapipe as mp
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

LANDMARK_SPEC   = mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=4, circle_radius=6)
CONNECTION_SPEC = mp_drawing.DrawingSpec(color=(0, 180, 255), thickness=4)

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
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_angle(a, b, c):
    """Interior angle at joint b using dot product (degrees)."""
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba, bc = a - b, c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm < 1e-8:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / norm, -1.0, 1.0))))


def reencode_h264(src: str, dst: str) -> tuple:
    """Re-encode to H.264 mp4 (CRF 18) so every browser can play it."""
    for candidate in ["ffmpeg", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if os.path.exists(candidate) or \
                subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0:
            ffmpeg = candidate
            break
    else:
        return False, "ffmpeg not found — add 'ffmpeg' to packages.txt in your repo root."

    r = subprocess.run(
        [ffmpeg, "-y", "-i", src,
         "-vcodec", "libx264", "-preset", "fast", "-crf", "18",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst],
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
        return None, None, None, None, 0, "cv2 could not open the file."

    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Keep native resolution; cap width at 1280 only for very large files
    if native_w > 1280:
        scale  = 1280 / native_w
        out_w  = 1280
        out_h  = int(native_h * scale)
        out_h += out_h % 2   # keep even
    else:
        out_w, out_h = native_w, native_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_out, fourcc, fps, (out_w, out_h))

    elbow_history, torque_history = [], []
    release_frame      = None
    frame_index        = 0
    landmarks_detected = 0

    bar = st.progress(0, text="Analysing video…")

    # ── model_complexity=1  ← CRITICAL: bundled lite model, no download needed
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if native_w != out_w:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = pose.process(rgb)
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
                    return [lm[i].x * out_w, lm[i].y * out_h]

                # Check both arms — use whichever is more extended
                r_elbow = calculate_angle(pt(12), pt(14), pt(16))
                l_elbow = calculate_angle(pt(11), pt(13), pt(15))
                elbow_angle   = max(r_elbow, l_elbow)
                side          = "R" if r_elbow >= l_elbow else "L"

                if side == "R":
                    hip_angle = calculate_angle(pt(12), pt(24), pt(26))
                else:
                    hip_angle = calculate_angle(pt(11), pt(23), pt(25))
                torque = abs(elbow_angle - hip_angle)

                elbow_history.append(elbow_angle)
                torque_history.append(torque)

                if elbow_angle > 120 and release_frame is None:
                    release_frame = frame_index

                # HUD — semi-transparent background strip
                font   = cv2.FONT_HERSHEY_SIMPLEX
                fsc    = max(0.6, out_w / 1600)
                thick  = max(2, int(fsc * 2.5))
                pad    = int(20 * fsc)
                strip_h = int(150 * fsc)

                overlay = annotated.copy()
                cv2.rectangle(overlay, (0, 0), (int(out_w * 0.45), strip_h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0, annotated)

                cv2.putText(annotated, f"Elbow ({side}): {int(elbow_angle)}",
                            (pad, int(45 * fsc)),  font, fsc,        (0, 255, 120),  thick)
                cv2.putText(annotated, f"Hip Torque: {int(torque)}",
                            (pad, int(88 * fsc)),  font, fsc,        (0, 200, 255),  thick)
                cv2.putText(annotated, f"Frame {frame_index}/{total}",
                            (pad, int(126 * fsc)), font, fsc * 0.75, (180, 180, 180), max(1, thick - 1))

            writer.write(annotated)
            frame_index += 1

            if total > 0:
                bar.progress(min(frame_index / total, 1.0),
                             text=f"Frame {frame_index} / {total}")

    cap.release()
    writer.release()
    bar.empty()

    return elbow_history, torque_history, release_frame, frame_index, landmarks_detected, None


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

    # ── Pose detection ────────────────────────────────────────────────────────
    elbow_history, torque_history, release_frame, total_frames, lm_count, err = \
        process_video(input_path, raw_path)

    if elbow_history is None:
        st.error(f"Could not open video: {err}")
        st.stop()

    detect_pct = int(lm_count / max(total_frames, 1) * 100)
    if detect_pct >= 60:
        st.success(f"✅  Analysis complete — {total_frames} frames, pose detected in {detect_pct}% of frames.")
    elif detect_pct > 0:
        st.warning(f"⚠️  Pose detected in only {detect_pct}% of frames. "
                   "Try better lighting or ensure the full body is in frame.")
    else:
        st.error("❌  No pose landmarks detected. Ensure the athlete is clearly visible, "
                 "well-lit, and the full body is in frame.")

    # ── Re-encode for browser playback ───────────────────────────────────────
    with st.spinner("Encoding video for playback…"):
        ok, enc_err = reencode_h264(raw_path, web_path)

    play_path = web_path if ok else raw_path
    if not ok:
        st.warning(f"Re-encode failed: `{enc_err}`  — make sure `packages.txt` contains `ffmpeg`.")

    with open(play_path, "rb") as f:
        st.video(f.read())

    # ── Release frame ─────────────────────────────────────────────────────────
    if release_frame is not None:
        st.info(f"🎯  Estimated Release Frame: **{release_frame}**")
    elif lm_count > 0:
        st.warning("Release frame not detected — elbow stayed below 120°. "
                   "Film from the side so arm extension is clearly visible.")

    # ── Metrics & feedback ────────────────────────────────────────────────────
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

