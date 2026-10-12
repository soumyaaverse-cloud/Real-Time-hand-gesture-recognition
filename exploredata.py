import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

df = pd.read_csv("data/gestures.csv")
print(df["label"].value_counts().to_string())


X = df.drop(columns="label").values
coords = PCA(n_components=2).fit_transform(X)

plt.figure(figsize=(7, 6))
for name in df["label"].unique():
    mask = (df["label"] == name).values
    plt.scatter(coords[mask, 0], coords[mask, 1], s=6, label=name, alpha=0.6)
plt.legend()
plt.title("Gestures in 2D (PCA)")
plt.savefig("outputs/pca.png", dpi=150)
print("saved outputs/pca.png")