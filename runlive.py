import os
import time
from collections import Counter, deque

import cv2
import joblib
import numpy as np

from common import (extract_features, get_first_hand, make_hands_detector,
                    mp_draw, mp_hands)

MODEL_FILE = os.path.join("models", "gesture_model.joblib")

MIN_CONFIDENCE = 0.6   
SMOOTHING_WINDOW = 7  
MIN_AGREEMENT = 4     


def main():
    if not os.path.exists(MODEL_FILE):
        print("no model found, run collect_data.py and train.py first")
        return

    model = joblib.load(MODEL_FILE)
    hands = make_hands_detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("couldn't open the webcam")
        return

    recent = deque(maxlen=SMOOTHING_WINDOW)
    shown_label = "-"
    latencies = []
    prev_frame_time = time.perf_counter()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        
        start = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        landmarks, side = get_first_hand(results)

        guess, conf = None, 0.0
        if landmarks is not None:
            feats = extract_features(landmarks, side).reshape(1, -1)
            probs = model.predict_proba(feats)[0]
            best = int(np.argmax(probs))
            guess, conf = model.classes_[best], float(probs[best])
        latencies.append((time.perf_counter() - start) * 1000)

        
        if guess is not None and conf >= MIN_CONFIDENCE:
            recent.append(guess)
        else:
            recent.append(None)

        votes = Counter(g for g in recent if g is not None)
        if votes:
            top, n = votes.most_common(1)[0]
            if n >= MIN_AGREEMENT:
                shown_label = top
        if landmarks is None:
            shown_label = "no hand"
            recent.clear()

        if landmarks is not None:
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_frame_time, 1e-6))
        prev_frame_time = now

        cv2.putText(frame, shown_label, (10, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, (0, 255, 0), 3)
        cv2.putText(frame, f"conf {conf:.2f}   {latencies[-1]:.0f} ms   {fps:.0f} fps",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("gesture recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if latencies:
        lat = np.array(latencies[10:] or latencies)  
        print(f"frames: {len(lat)}")
        print(f"avg latency: {lat.mean():.1f} ms")
        print(f"p95 latency: {np.percentile(lat, 95):.1f} ms")


if __name__ == "__main__":
    main()