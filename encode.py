import os
import cv2
import pickle
import numpy as np
from logger import get_logger
from insightface.app import FaceAnalysis

logger = get_logger("encode")

def encode_faces(dataset_dir="dataset"):
    os.makedirs("outputs/encodings", exist_ok=True)

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))

    encodings = {}
    mean_encodings = {}

    for person in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person)
        if not os.path.isdir(person_dir):
            continue

        embeddings = []
        for fname in os.listdir(person_dir):
            img_path = os.path.join(person_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Failed to read {img_path}")
                continue

            faces = app.get(img)
            if not faces:
                logger.warning(f"No face found in {img_path}")
                continue

            embeddings.append(faces[0].embedding)

        if embeddings:
            encodings[person] = embeddings
            mean_encodings[person] = np.mean(embeddings, axis=0)
            logger.info(f"Encoded {len(embeddings)} images for {person}")
        else:
            logger.warning(f"No embeddings for {person}")

    with open("outputs/encodings/encodings.pkl", "wb") as f:
        pickle.dump(encodings, f)
    with open("outputs/encodings/mean_encodings.pkl", "wb") as f:
        pickle.dump(mean_encodings, f)

    logger.info("Saved encodings to outputs/encodings/")

if __name__ == "__main__":
    encode_faces()
