"""
tests/test_face_engine.py — Unit tests for the DeepFace-based engine.
Run with: pytest tests/ -v
"""

import json
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from components.face_engine import (
    encoding_to_str, str_to_encoding,
    cosine_similarity, similarity_to_confidence,
    EncodingCache,
)


# ─── Encoding serialisation ───────────────────────────────────────────────────

def test_encoding_roundtrip():
    """Embedding → JSON string → numpy array must be lossless."""
    enc = np.random.rand(512).astype(np.float64)   # ArcFace uses 512-dim
    assert np.allclose(enc, str_to_encoding(encoding_to_str(enc)), rtol=1e-6)


def test_encoding_is_valid_json():
    enc = np.zeros(512)
    parsed = json.loads(encoding_to_str(enc))
    assert len(parsed) == 512


# ─── Cosine similarity ────────────────────────────────────────────────────────

def test_cosine_identical():
    v = np.random.rand(512)
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_cosine_zero_vector():
    a = np.zeros(512)
    b = np.random.rand(512)
    assert cosine_similarity(a, b) == 0.0


# ─── Confidence scoring ───────────────────────────────────────────────────────

def test_confidence_perfect():
    assert similarity_to_confidence(1.0) == 100.0


def test_confidence_zero():
    assert similarity_to_confidence(0.0) == 0.0


def test_confidence_half():
    assert abs(similarity_to_confidence(0.5) - 50.0) < 0.1


def test_confidence_clamp():
    assert similarity_to_confidence(-0.5) == 0.0


# ─── EncodingCache ────────────────────────────────────────────────────────────

def _make_cache():
    c = EncodingCache()
    c.add("Alice", np.random.rand(512), person_id=1)
    c.add("Bob",   np.random.rand(512), person_id=2)
    return c


def test_cache_add():
    c = _make_cache()
    assert c.size == 2
    assert "Alice" in c.names
    assert "Bob" in c.names


def test_cache_remove():
    c = _make_cache()
    c.remove(1)
    assert c.size == 1
    assert "Alice" not in c.names


def test_cache_load_from_db():
    c = EncodingCache()
    enc = np.random.rand(512)
    persons = [
        {"id": 1, "name": "Carol", "encoding": encoding_to_str(enc)},
        {"id": 2, "name": "Dave",  "encoding": None},           # skipped
        {"id": 3, "name": "Eve",   "encoding": "not-json"},     # skipped
    ]
    c.load_from_db(persons)
    assert c.size == 1
    assert c.names[0] == "Carol"


def test_cache_load_preserves_ids():
    c = EncodingCache()
    enc = np.random.rand(512)
    c.load_from_db([{"id": 42, "name": "Frank", "encoding": encoding_to_str(enc)}])
    assert c.ids[0] == 42
    