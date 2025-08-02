import streamlit as st
import face_recognition
import os
from PIL import Image, ImageDraw
import numpy as np

# Set title
st.title("Face Recognition App 🎯")
st.write("Upload an image and we’ll try to recognize known faces!")

# Path to folder containing known faces
KNOWN_FACES_DIR = "known_faces"

# Load known face encodings
known_face_encodings = []
known_face_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)
        else:
            st.warning(f"No face found in {filename}, skipping.")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load uploaded image
    uploaded_image = Image.open(uploaded_file)
    image_np = np.array(uploaded_image)

    # Find all faces and face encodings in the uploaded image
    face_locations = face_recognition.face_locations(image_np)
    face_encodings = face_recognition.face_encodings(image_np, face_locations)

    # Create a PIL image to draw on
    draw_image = uploaded_image.copy()
    draw = ImageDraw.Draw(draw_image)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw box and name
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=3)
        draw.text((left + 6, bottom + 5), name, fill=(255, 255, 255))

    # Show result
    st.image(draw_image, caption="Detected Faces")

    if not face_locations:
        st.warning("No face detected in uploaded image.")
