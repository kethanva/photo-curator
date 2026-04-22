"""
Facial sentiment scoring using MediaPipe Face Mesh.

Detects per-face:
  - Eye openness via Eye Aspect Ratio (EAR)
  - Smile presence via mouth-corner elevation relative to upper lip

Returns a 0–1 score where:
  1.0 = all faces smiling with eyes wide open
  0.5 = neutral expression or no faces detected
  0.0 = eyes closed / sad expression

Install:
    pip install mediapipe
"""

from __future__ import annotations

import os
import numpy as np
from PIL import Image

# Silence MediaPipe's C++ INFO/WARNING logs (GL context, feedback manager, etc.)
# These are harmless runtime notes from the Metal GPU backend on Apple Silicon.
os.environ.setdefault("GLOG_minloglevel", "2")       # suppress INFO + WARNING
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # suppress TF/XLA noise

# ---------------------------------------------------------------------------
# MediaPipe landmark indices (468-point Face Mesh)
# ---------------------------------------------------------------------------

# Eye landmark indices: [outer, top1, top2, inner, bot1, bot2]
_LEFT_EYE = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Mouth landmarks
_MOUTH_LEFT_CORNER = 61
_MOUTH_RIGHT_CORNER = 291
_MOUTH_UPPER_LIP = 13    # centre of upper lip
_MOUTH_LOWER_LIP = 14    # centre of lower lip

# EAR threshold: >0.20 = open, <0.15 = closed
_EAR_OPEN_THRESHOLD = 0.20

# Smile sensitivity — positive = corners above lip centre (smiling)
_SMILE_SENSITIVITY = 25.0


def _ear(lm, indices) -> float:
    """
    Eye Aspect Ratio.
    EAR = (|p1-p5| + |p2-p4|) / (2 * |p0-p3|)
    p0=outer, p3=inner, p1/p2=top, p4/p5=bottom
    """
    pts = [(lm[i].x, lm[i].y) for i in indices]
    v1 = abs(pts[1][1] - pts[5][1])
    v2 = abs(pts[2][1] - pts[4][1])
    h = abs(pts[0][0] - pts[3][0])
    return (v1 + v2) / (2.0 * h + 1e-8)


def _smile_score(lm) -> float:
    """
    Smile score [0, 1].
    Corners elevated above upper lip centre = smiling (positive value).
    """
    corner_y = (lm[_MOUTH_LEFT_CORNER].y + lm[_MOUTH_RIGHT_CORNER].y) / 2.0
    lip_y = lm[_MOUTH_UPPER_LIP].y
    # In normalised coords, y increases downward.
    # Smiling → corners have smaller y (raised) → lip_y > corner_y
    raw = (lip_y - corner_y) * _SMILE_SENSITIVITY + 0.5
    return float(np.clip(raw, 0.0, 1.0))


def score_image(img: Image.Image) -> float:
    """
    Compute facial sentiment score for an image.

    Returns:
        0.5 if no faces detected (neutral / unknown).
        0–1 based on eye openness and smile for the primary face.
    """
    try:
        import mediapipe as mp

        rgb = np.array(img.convert("RGB"))
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        )

        with face_mesh:
            results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return 0.5  # No face — neutral score

        face_scores = []
        for face in results.multi_face_landmarks:
            lm = face.landmark

            left_ear = _ear(lm, _LEFT_EYE)
            right_ear = _ear(lm, _RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0

            # Eye openness: 0 = closed, 1 = fully open
            eye_open = float(np.clip(avg_ear / _EAR_OPEN_THRESHOLD, 0.0, 1.0))

            smile = _smile_score(lm)

            # Weighted combination: eyes matter more (blinks vs smiles)
            face_score = 0.55 * eye_open + 0.45 * smile
            face_scores.append(face_score)

        # Use best face score (most expressive / primary subject)
        return float(max(face_scores))

    except ImportError:
        # MediaPipe not installed — return neutral score silently
        return 0.5
    except Exception:
        return 0.5
