import csv
import os
import sys
from collections import Counter

import cv2

from common import (GESTURES, N_FEATURES, extract_features, get_first_hand,
                    make_hands_detector, mp_draw, mp_hands)


DATA_FILE = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "gestures.csv")
TARGET_PER_GESTURE = 400  


def count_existing():
    """Rows per gesture already saved, so we can keep adding to the file."""
    counts = Counter()
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  
            for row in reader:
                if row:
                    counts[row[0]] += 1
    return counts


def main():
    os.makedirs("data", exist_ok=True)

    file_is_new = not os.path.exists(DATA_FILE)
    csv_file = open(DATA_FILE, "a", newline="")
    writer = csv.writer(csv_file)
    if file_is_new:
        writer.writerow(["label"] + [f"f{i}" for i in range(N_FEATURES)])

    totals = count_existing()
    session = Counter()
    recording = None  

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("couldn't open the webcam - is another app using it?")
        return

    hands = make_hands_detector()
    print("ready. press 1-5 to record, q to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        landmarks, side = get_first_hand(results)
        if landmarks is not None:
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            if recording is not None:
                writer.writerow([recording] + extract_features(landmarks, side).tolist())
                session[recording] += 1
                totals[recording] += 1

                if session[recording] >= TARGET_PER_GESTURE:
                    print(f"{recording}: hit {TARGET_PER_GESTURE} this session, stopping")
                    recording = None

        # little HUD so you know what's going on
        status = f"recording: {recording}" if recording else "idle"
        color = (0, 0, 255) if recording else (200, 200, 200)
        cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        for i, name in enumerate(GESTURES):
            line = f"{i + 1}. {name}: {totals[name]}"
            cv2.putText(frame, line, (10, 60 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if landmarks is None:
            cv2.putText(frame, "no hand", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        cv2.imshow("collect", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if ord("1") <= key <= ord("0") + len(GESTURES):
            name = GESTURES[key - ord("1")]
            recording = None if recording == name else name  # same key again = stop

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    print("saved. totals:", dict(totals))


if __name__ == "__main__":
    main()