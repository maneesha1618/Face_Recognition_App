"""
1_Register_Face.py — Enroll new faces into the system.
Uses DeepFace (ArcFace) for embedding — no dlib required.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from PIL import Image

from components.database import init_db, add_person
from components.face_engine import (
    cache, encode_face_from_image, encoding_to_str, save_face_image
)
from components.utils import page_config, inject_global_css

page_config("Register Face", "📸")
inject_global_css()
init_db()

st.title("📸 Register New Face")
st.caption("Enroll a person so FaceVault can recognise them. Uses ArcFace for high accuracy.")
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("register_form", clear_on_submit=True):
    col_info, col_img = st.columns([1, 1])

    with col_info:
        st.subheader("Personal Details")
        name        = st.text_input("Full Name *", placeholder="e.g. Maneesha Raj")
        employee_id = st.text_input("Employee / Student ID", placeholder="Optional")
        department  = st.text_input("Department", placeholder="e.g. Engineering")
        email       = st.text_input("Email", placeholder="optional@email.com")

    with col_img:
        st.subheader("Face Image")
        source = st.radio(
            "Image source", ["Upload a photo", "Use webcam"],
            horizontal=True,
        )

        uploaded_file = None
        camera_photo  = None

        if source == "Upload a photo":
            uploaded_file = st.file_uploader(
                "Choose an image", type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )
        else:
            camera_photo = st.camera_input("Take a snapshot")

    submitted = st.form_submit_button("✅ Register Face", use_container_width=True)


# ── Processing ────────────────────────────────────────────────────────────────
if submitted:
    if not name.strip():
        st.error("❌ Full name is required.")
        st.stop()

    raw_image = None
    if uploaded_file:
        raw_image = Image.open(uploaded_file).convert("RGB")
    elif camera_photo:
        raw_image = Image.open(camera_photo).convert("RGB")
    else:
        st.error("❌ Please provide a face image.")
        st.stop()

    with st.spinner("🔍 Detecting face and generating ArcFace embedding… (first run downloads model)"):
        encoding = encode_face_from_image(raw_image)

    if encoding is None:
        st.error(
            "❌ No face detected. Please use a clear, front-facing photo with good lighting."
        )
        st.stop()

    with st.spinner("💾 Saving to database…"):
        img_path = save_face_image(raw_image, name.strip())
        enc_str  = encoding_to_str(encoding)

        person = add_person(
            name=name.strip(),
            face_image=img_path,
            encoding=enc_str,
            employee_id=employee_id.strip() or None,
            department=department.strip() or None,
            email=email.strip() or None,
        )

        cache.add(name.strip(), encoding, person.id)
        st.session_state["cache_loaded"] = False

    st.success(f"🎉 **{name.strip()}** registered successfully! (ID: {person.id})")
    st.balloons()

    col_prev, _ = st.columns([1, 2])
    with col_prev:
        st.image(raw_image, caption=name.strip(), width=220)


# ── Tips ──────────────────────────────────────────────────────────────────────
with st.expander("💡 Tips for best results"):
    st.markdown("""
    - **Lighting**: Even, front-facing light — avoid backlight or harsh shadows.
    - **Angle**: Straight-on is best. Slight angles are fine.
    - **Resolution**: At least 200×200 px on the face region.
    - **Expression**: Neutral or slight smile works best for ArcFace.
    - **Multiple enrollments**: Register the same person 2–3 times (different lighting/angles)
      for significantly better real-world accuracy.
    """)