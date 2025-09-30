import pickle
import os
from sklearn.neighbors import KNeighborsClassifier
from logger import get_logger
import numpy as np

logger = get_logger("train")

def train_classifier(enc_path="outputs/encodings/encodings.pkl", out_path="outputs/classifiers/classifier.pkl"):
    os.makedirs("outputs/classifiers", exist_ok=True)

    with open(enc_path, "rb") as f:
        encodings = pickle.load(f)

    X, y = [], []
    for person, emb_list in encodings.items():
        for emb in emb_list:
            X.append(emb)
            y.append(person)

    if not X:
        logger.error("No embeddings found. Run encode.py first.")
        return

    clf = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    clf.fit(np.array(X), y)

    with open(out_path, "wb") as f:
        pickle.dump(clf, f)

    logger.info(f"Saved classifier to {out_path}")

if __name__ == "__main__":
    train_classifier()
