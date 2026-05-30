"""
2_Recognize_Face.py — Recognise faces from an uploaded image or webcam snapshot.
Uses DeepFace (ArcFace) backend — no dlib required.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from PIL import Image

from components.database import init_db, get_all_persons, log_attendance, already_logged_today
from components.face_engine import (
    cache, recognize_faces_in_image, draw_results_on_image
)
from components.utils import page_config, inject_global_css, confidence_badge

page_config("Recognize Face", "🔍")
inject_global_css()
init_db()

# ── Reload cache if needed ─────────────────────────────────────────────────────
if not st.session_state.get("cache_loaded") or cache.size == 0:
    persons = get_all_persons()
    cache.load_from_db(persons)
    st.session_state["cache_loaded"] = True

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 Face Recognition")
st.caption("Upload a photo or use your webcam to identify registered faces.")
st.divider()

if cache.size == 0:
    st.warning(
        "⚠️ No faces are registered yet. "
        "Go to **Register Face** first to enrol at least one person."
    )
    st.stop()

st.info(f"🧠 ArcFace engine ready · **{cache.size}** face(s) in memory")

# ── Mode select ───────────────────────────────────────────────────────────────
mode = st.radio(
    "Select input mode",
    ["📁 Upload Image", "📷 Webcam Snapshot"],
    horizontal=True,
)

source_image: Image.Image | None = None

if mode == "📁 Upload Image":
    uploaded = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"],
    )
    if uploaded:
        source_image = Image.open(uploaded).convert("RGB")
else:
    snap = st.camera_input("Take a photo")
    if snap:
        source_image = Image.open(snap).convert("RGB")

# ── Run recognition ───────────────────────────────────────────────────────────
if source_image is not None:

    with st.spinner("🔎 Running ArcFace recognition… (first run downloads model ~100 MB)"):
        results      = recognize_faces_in_image(source_image)
        annotated    = draw_results_on_image(source_image, results)

    col_img, col_results = st.columns([1, 1])

    with col_img:
        st.subheader("📸 Result")
        st.image(annotated, use_container_width=True)

    with col_results:
        st.subheader(f"🎯 Detected {len(results)} face(s)")

        if not results:
            st.info("No faces found in this image.")
        else:
            for r in results:
                name  = r["name"]
                conf  = r["confidence"]
                match = r["match"]

                if match:
                    st.markdown(
                        f"### ✅ {name}\n"
                        f"{confidence_badge(conf)}",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("### ❓ Unknown Person")
                st.divider()

        # ── Log attendance ─────────────────────────────────────────────────
        st.subheader("📋 Log Attendance")
        auto_log = st.toggle("Only log recognised faces", value=True)

        if st.button("💾 Save to Attendance Log", use_container_width=True):
            logged, skipped = 0, 0
            for r in results:
                name = r["name"]
                if auto_log and not r["match"]:
                    continue
                if already_logged_today(name):
                    skipped += 1
                    continue
                log_attendance(
                    person_name=name,
                    confidence=r["confidence"],
                    person_id=r.get("person_id"),
                    status="Present" if r["match"] else "Unknown",
                )
                logged += 1

            if logged:
                st.success(f"✅ Logged {logged} record(s).")
            if skipped:
                st.info(f"ℹ️ {skipped} already logged today — skipped.")

# ── First-run note ─────────────────────────────────────────────────────────────
with st.expander("ℹ️ First-run note"):
    st.markdown("""
    **On first use**, DeepFace will automatically download the ArcFace model weights (~100 MB).
    This happens once and is cached in `~/.deepface/`. Subsequent runs are instant.

    **Models available** (set `DEEPFACE_MODEL` in `.env`):
    | Model | Size | Speed | Accuracy |
    |---|---|---|---|
    | `ArcFace` | ~100 MB | Medium | ⭐⭐⭐⭐⭐ |
    | `Facenet512` | ~90 MB | Fast | ⭐⭐⭐⭐ |
    | `VGG-Face` | ~500 MB | Slow | ⭐⭐⭐⭐ |
    | `SFace` | ~40 MB | Fastest | ⭐⭐⭐ |
    """)