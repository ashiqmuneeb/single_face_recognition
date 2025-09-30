import cv2
import os
import pickle
import argparse
import numpy as np
from logger import get_logger
from insightface.app import FaceAnalysis
from sklearn.neighbors import KNeighborsClassifier

logger = get_logger("recognize")

def recognize_faces(mean_path="outputs/encodings/mean_encodings.pkl",
                    classifier_path=None,
                    threshold=0.40):
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))

    with open(mean_path, "rb") as f:
        mean_encodings = pickle.load(f)
    logger.info(f"Loaded mean encodings for {len(mean_encodings)} persons")

    clf = None
    if classifier_path and os.path.exists(classifier_path):
        with open(classifier_path, "rb") as f:
            clf = pickle.load(f)
        logger.info("Loaded classifier")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Webcam not found")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        for face in faces:
            emb = face.embedding

            name, conf = "Unknown", 0.0

            if clf:  # use classifier
                pred = clf.predict([emb])[0]
                dist, _ = clf.kneighbors([emb], n_neighbors=1, return_distance=True)
                if dist[0][0] < threshold:
                    name, conf = pred, float(dist[0][0])
            else:  # use cosine similarity with mean encodings
                sims = {p: np.dot(emb, e) / (np.linalg.norm(emb) * np.linalg.norm(e))
                        for p, e in mean_encodings.items()}
                name, conf = max(sims.items(), key=lambda x: x[1])
                if conf < threshold:
                    name = "Unknown"

            box = face.bbox.astype(int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({conf:.2f})", (box[0], box[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Recognition stopped")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mean", default="outputs/encodings/mean_encodings.pkl", help="Path to mean encodings")
    parser.add_argument("--classifier", default=None, help="Path to trained classifier")
    parser.add_argument("--threshold", type=float, default=0.40, help="Threshold for recognition")

    args = parser.parse_args()
    recognize_faces(args.mean, args.classifier, args.threshold)
