import sys

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

path = sys.argv[1] if len(sys.argv) > 1 else "data/test_session.csv"

model = joblib.load("models/gesture_model.joblib")
df = pd.read_csv(path)
X = df.drop(columns="label").values
y = df["label"].values

preds = model.predict(X)
print(f"accuracy on fresh session: {accuracy_score(y, preds):.3f}\n")
print(classification_report(y, preds))