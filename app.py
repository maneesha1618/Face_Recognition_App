import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
from face_recognition_engine import load_known_faces, recognize_faces

st.set_page_config(page_title="Face Recognition", layout="centered")
st.title(" Face Recognition App ")

# Function to reload known faces
@st.cache_resource
def load_faces():
    return load_known_faces("known_faces")

known_encodings, known_names = load_faces()

# Tabs for face recognition and new face registration
tab1, tab2 = st.tabs(["🔍 Recognize Face", "➕ Register New Face"])

with tab1:
    st.subheader("Detect a Known Face")

    input_method = st.radio("Select input method:", ("Upload Image", "Use Webcam"))

    image = None

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")

    elif input_method == "Use Webcam":
        webcam_capture = st.camera_input("📸 Capture an image")
        if webcam_capture:
            image = Image.open(webcam_capture).convert("RGB")

    if image is not None:
        st.subheader("Input Image:")
        st.image(image, use_column_width=True)

        img_array = np.array(image)

        # Run recognition
        result_img, detected_names = recognize_faces(img_array, known_encodings, known_names)

        display_img = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        st.subheader("Result:")
        st.image(display_img, caption="🧠 Processed with Recognition", use_column_width=True)

        if detected_names:
            st.success(f"✅ Faces Detected: {', '.join(set(detected_names))}")
        else:
            st.info("😐 No known faces detected.")

with tab2:
    st.subheader("Register a New Face")

    name = st.text_input("Enter the name of the person:")
    new_face_capture = st.camera_input("📸 Capture face photo")

    if new_face_capture and name:
        img = Image.open(new_face_capture).convert("RGB")
        save_path = os.path.join("known_faces", f"{name.replace(' ', '_')}.jpg")
        img.save(save_path)
        st.success(f"🎉 Face of '{name}' saved successfully!")

        # Reload known faces after saving
        known_encodings, known_names = load_known_faces("known_faces")
        st.info("✅ Updated known faces database. You can now recognize this person.")
    elif new_face_capture and not name:
        st.warning("⚠️ Please enter a name before capturing.")
