import os

import joblib
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_FILE = os.path.join("data", "gestures.csv")
MODEL_FILE = os.path.join("models", "gesture_model.joblib")
CM_FILE = os.path.join("outputs", "confusion_matrix.png")


def main():
    df = pd.read_csv(DATA_FILE)
    print("rows per gesture:")
    print(df["label"].value_counts().to_string(), "\n")

    X = df.drop(columns="label").values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    candidates = {
        "knn": KNeighborsClassifier(n_neighbors=5),
        "svm": make_pipeline(StandardScaler(), SVC(C=10, probability=True)),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    
    best_name, best_score = None, 0.0
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"{name:14s} cv accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
        if scores.mean() > best_score:
            best_name, best_score = name, scores.mean()

    print(f"\ngoing with {best_name}")
    model = candidates[best_name]
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"test accuracy: {accuracy_score(y_test, preds):.3f}\n")
    print(classification_report(y_test, preds))

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds, ax=ax, xticks_rotation=30, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Confusion matrix ({best_name})")
    fig.tight_layout()
    fig.savefig(CM_FILE, dpi=150)
    print("confusion matrix saved to", CM_FILE)

    joblib.dump(model, MODEL_FILE)
    print("model saved to", MODEL_FILE)


if __name__ == "__main__":
    main()