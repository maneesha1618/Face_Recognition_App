import os
import face_recognition
import cv2
import numpy as np
from typing import List, Tuple

def load_known_faces(known_faces_dir: str) -> Tuple[List[np.ndarray], List[str]]:
    known_encodings = []
    known_names = []

    for filename in os.listdir(known_faces_dir):
        filepath = os.path.join(known_faces_dir, filename)
        if not filename.lower().endswith(('jpg', 'jpeg', 'png')):
            continue

        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0].replace('_', ' ').title()
            known_names.append(name)
        else:
            print(f"[WARNING] No face found in image {filename}, skipping.")

    return known_encodings, known_names

def recognize_faces(image: np.ndarray, known_encodings: List[np.ndarray], known_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    # Convert to RGB if image is BGR (OpenCV format)
    if image.shape[-1] == 3:
        rgb_image = image[:, :, ::-1]  # BGR to RGB
    else:
        rgb_image = image

    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    detected_names = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
            else:
                name = "Unknown"
        else:
            name = "Unknown"

        detected_names.append(name)

        # Draw rectangle around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw label below the image instead of over the face
        label_y = bottom + 20 if bottom + 20 < image.shape[0] else bottom - 10
        cv2.putText(image, name, (left, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return image, detected_names

