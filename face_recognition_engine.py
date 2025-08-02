import os
import cv2
import face_recognition

def load_known_faces(known_faces_dir):
    known_encodings = []
    known_names = []
    
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])
    return known_encodings, known_names

def recognize_faces(image, known_encodings, known_names):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    detected_names = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]
            detected_names.append(name)
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return image, detected_names
