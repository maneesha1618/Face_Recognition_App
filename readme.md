## Face Recognition App

A simple web application for detecting and recognizing known faces using images or your webcam. Built using **Streamlit, OpenCV, and face_recognition**.

---

### 📸 Features

* Upload an image or capture one from your webcam.
* Detect and recognize faces from a known set.
* Display results with names and bounding boxes on the image.

---

### 🛠️ Tech Stack

* Python 3.8 
* Streamli 
* OpenC 
* face_recognition
* Numpy
* PIL

---

### 📁 Folder Structure

```
face_recognition_app/
│
├── known_faces/                # Folder containing known images (named files)
│   ├── john_doe.jpg
│   └── jane_smith.png
│
├── face_recognition_engine.py  # Core logic for loading & recognizing faces
├── app.py                      # Streamlit app file (main UI)
├── requirements.txt
└── README.md
```

---

### 🚀 Getting Started

#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/face-recognition-app.git
cd face-recognition-app
```

#### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

#### 3. Install Requirements

```bash
pip install -r requirements.txt
```

#### 4. Add Known Faces

Place images of known people inside the `known_faces/` directory.
**Important:** The filename (without extension) will be used as the person’s name.

Example:

```
known_faces/
├── john_doe.jpg   →  Name detected: "john_doe"
├── alice.png      →  Name detected: "alice"
```

#### 5. Run the App

```bash
streamlit run app.py
```

Then open the link shown in your terminal (usually `http://localhost:8501`) in your browser.

---

### ✅ Notes

* For webcam support, Streamlit's `st.camera_input()` is used. It works in browsers that support camera access.
* Recognition is done using face encodings from the `face_recognition` library.

---

### 📦 Example Output

* Uploaded image is displayed.
* Bounding boxes and names appear on recognized faces.
* Messages displayed for recognition success or failure.

---

### 🔒 Disclaimer

This project is for educational and demo purposes. Do not use it in production without proper privacy and security measures.
