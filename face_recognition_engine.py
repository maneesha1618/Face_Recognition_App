import os
import face_recognition
import numpy as np
import cv2
from typing import List, Tuple

def load_known_faces(known_faces_dir: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load known face encodings and names from a given folder.
    """
    known_encodings = []
    known_names = []

    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)

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

def recognize_faces(image: np.ndarray, known_encodings: List[np.ndarray], known_names: List[str], tolerance=0.45) -> Tuple[np.ndarray, List[str]]:
    """
    Detect and recognize faces in an image.
    """
    rgb_image = image[:, :, ::-1] if image.shape[-1] == 3 else image

    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    detected_names = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        name = "Unknown"
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index] and face_distances[best_match_index] < tolerance:
                name = known_names[best_match_index]

        detected_names.append(name)

        # Draw rectangle and label
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return image, detected_names
