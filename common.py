import numpy as np
import mediapipe as mp

#
GESTURES = ["fist", "open_palm", "thumbs_up", "peace", "point"]

N_LANDMARKS = 21
N_FEATURES = N_LANDMARKS * 2  # x and y only, z is too noisy from a webcam

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def make_hands_detector():
   
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )


def extract_features(hand_landmarks, handedness_label):
    """21 landmarks -> 42 numbers that ignore hand position and size.

    1. shift so the wrist is (0, 0)        -> position independent
    2. divide by the biggest distance      -> size independent
    3. mirror left hands                   -> one model for both hands
    """
    pts = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32)

    pts = pts - pts[0]

    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale

    if handedness_label == "Left":
        pts[:, 0] = -pts[:, 0]

    return pts.flatten()


def get_first_hand(results):
    """Returns (landmarks, 'Left'/'Right') or (None, None) if no hand found."""
    if not results.multi_hand_landmarks:
        return None, None
    landmarks = results.multi_hand_landmarks[0]
    label = results.multi_handedness[0].classification[0].label
    return landmarks, label