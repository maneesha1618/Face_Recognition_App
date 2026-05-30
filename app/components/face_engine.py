"""
face_engine.py — Core face recognition logic (DeepFace backend)
Uses DeepFace with ArcFace model — no dlib/CMake required.
Pure pip install on Windows, Linux, and macOS.
"""

import json
import logging
import os
import time
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

MIN_CONF   = float(os.getenv("MIN_CONFIDENCE", "60"))
FACES_DIR  = Path(os.getenv("FACES_DIR", "data/known_faces"))
UNKNOWN    = os.getenv("UNKNOWN_LABEL", "Unknown")

# DeepFace model — ArcFace is best accuracy; Facenet512 is fastest
DEEPFACE_MODEL    = os.getenv("DEEPFACE_MODEL", "ArcFace")
DEEPFACE_DETECTOR = os.getenv("DEEPFACE_DETECTOR", "opencv")   # opencv | retinaface | mtcnn
DEEPFACE_METRIC   = os.getenv("DEEPFACE_METRIC", "cosine")     # cosine | euclidean_l2

# Cosine threshold: lower = stricter (0.0 = identical, 1.0 = totally different)
COSINE_THRESHOLD  = float(os.getenv("COSINE_THRESHOLD", "0.30"))

FACES_DIR.mkdir(parents=True, exist_ok=True)


# ─── Lazy DeepFace import (avoids slow TF load at module level) ───────────────

def _deepface():
    from deepface import DeepFace  # noqa: PLC0415
    return DeepFace


# ─── Encoding helpers ─────────────────────────────────────────────────────────

def encoding_to_str(enc: np.ndarray) -> str:
    """Serialise a face embedding to a JSON string for DB storage."""
    return json.dumps(enc.tolist())


def str_to_encoding(s: str) -> np.ndarray:
    """Deserialise a face embedding from a JSON string."""
    return np.array(json.loads(s))


# ─── Image helpers ────────────────────────────────────────────────────────────

def pil_to_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = pil_to_rgb(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _save_temp(img: Image.Image) -> str:
    """Save PIL image to a temp file and return the path (DeepFace needs a path)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.convert("RGB").save(tmp.name, "JPEG", quality=95)
    return tmp.name


# ─── Encoding ─────────────────────────────────────────────────────────────────

def encode_face_from_image(img: Image.Image) -> Optional[np.ndarray]:
    """
    Return the ArcFace embedding for the first face found in a PIL image.
    Returns None if no face is detected.
    """
    tmp_path = _save_temp(img)
    try:
        result = _deepface().represent(
            img_path=tmp_path,
            model_name=DEEPFACE_MODEL,
            detector_backend=DEEPFACE_DETECTOR,
            enforce_detection=True,
            align=True,
        )
        if result:
            return np.array(result[0]["embedding"])
        return None
    except Exception as exc:
        logger.warning("No face detected or encoding failed: %s", exc)
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def encode_face_from_path(image_path: str) -> Optional[np.ndarray]:
    try:
        img = Image.open(image_path)
        return encode_face_from_image(img)
    except Exception as exc:
        logger.error("Failed to load %s: %s", image_path, exc)
        return None


# ─── In-memory embedding cache ────────────────────────────────────────────────

class EncodingCache:
    """
    Holds all known face embeddings in RAM for fast cosine comparison.
    Rebuilt at app startup, hot-updated on register/delete.
    """

    def __init__(self):
        self.encodings: list[np.ndarray] = []
        self.names:     list[str]        = []
        self.ids:       list[Optional[int]] = []

    def load_from_db(self, persons: list[dict]):
        self.encodings.clear()
        self.names.clear()
        self.ids.clear()

        for p in persons:
            enc_str = p.get("encoding")
            if not enc_str:
                continue
            try:
                enc = str_to_encoding(enc_str)
                self.encodings.append(enc)
                self.names.append(p["name"])
                self.ids.append(p.get("id"))
            except Exception as exc:
                logger.warning("Bad encoding for %s: %s", p.get("name"), exc)

        logger.info("Cache loaded: %d face(s)", len(self.encodings))

    def add(self, name: str, encoding: np.ndarray, person_id: Optional[int] = None):
        self.encodings.append(encoding)
        self.names.append(name)
        self.ids.append(person_id)

    def remove(self, person_id: int):
        indices = [i for i, pid in enumerate(self.ids) if pid == person_id]
        for i in reversed(indices):
            self.encodings.pop(i)
            self.names.pop(i)
            self.ids.pop(i)

    @property
    def size(self) -> int:
        return len(self.encodings)


# Singleton — import this in every page
cache = EncodingCache()


# ─── Similarity / confidence ──────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity in [0, 1]. 1 = identical."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def similarity_to_confidence(similarity: float) -> float:
    """Map cosine similarity → 0–100 % confidence."""
    return round(max(0.0, similarity * 100), 1)


# ─── Recognition ──────────────────────────────────────────────────────────────

def recognize_faces_in_image(img: Image.Image) -> list[dict]:
    """
    Detect all faces in a PIL image and match each against the cache.

    Returns list of dicts:
        {
            "name":       str,
            "confidence": float,      # 0–100
            "region":     dict,       # x, y, w, h from DeepFace
            "match":      bool,
            "person_id":  int | None,
        }
    """
    if cache.size == 0:
        return []

    tmp_path = _save_temp(img)
    try:
        faces = _deepface().represent(
            img_path=tmp_path,
            model_name=DEEPFACE_MODEL,
            detector_backend=DEEPFACE_DETECTOR,
            enforce_detection=False,
            align=True,
        )
    except Exception as exc:
        logger.warning("DeepFace represent failed: %s", exc)
        return []
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    results = []
    for face_data in faces:
        query_enc = np.array(face_data["embedding"])
        region    = face_data.get("facial_area", {})

        # Compare against every cached embedding
        similarities = [cosine_similarity(query_enc, enc) for enc in cache.encodings]
        best_idx  = int(np.argmax(similarities))
        best_sim  = similarities[best_idx]
        confidence = similarity_to_confidence(best_sim)

        # Threshold check: cosine similarity must exceed (1 - COSINE_THRESHOLD)
        matched = best_sim >= (1.0 - COSINE_THRESHOLD) and confidence >= MIN_CONF

        results.append({
            "name":      cache.names[best_idx] if matched else UNKNOWN,
            "confidence": confidence if matched else 0.0,
            "region":    region,
            "match":     matched,
            "person_id": cache.ids[best_idx] if matched else None,
        })

    return results


def draw_results_on_image(img: Image.Image, results: list[dict]) -> Image.Image:
    """
    Draw bounding boxes and labels on a PIL image.
    Returns annotated PIL image.
    """
    frame = pil_to_bgr(img)

    for r in results:
        region = r.get("region", {})
        x = region.get("x", 0)
        y = region.get("y", 0)
        w = region.get("w", 0)
        h = region.get("h", 0)

        color = (0, 200, 80) if r["match"] else (0, 60, 220)
        label = f"{r['name']}  {r['confidence']:.0f}%" if r["match"] else UNKNOWN

        # Bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x, y - th - 12), (x + tw + 8, y), color, -1)

        # Label text
        cv2.putText(
            frame, label,
            (x + 4, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ─── Face image storage ───────────────────────────────────────────────────────

def save_face_image(img: Image.Image, name: str) -> str:
    safe_name = "".join(c if c.isalnum() else "_" for c in name)
    filename  = f"{safe_name}_{int(time.time())}.jpg"
    path      = FACES_DIR / filename
    img.convert("RGB").save(str(path), "JPEG", quality=90)
    return str(path)