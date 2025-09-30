Face Recognition Project

This project implements a face recognition pipeline using the InsightFace w600k_r50.onnx model. It allows capturing images, encoding faces, training a classifier, and recognizing faces in real time. The system supports single-person datasets, auto/manual capture, and logs all activities.



⚡ Features

Capture images from webcam (auto/manual)

Encode faces using InsightFace w600k_r50.onnx model

Train KNN classifier on embeddings

Real-time recognition with “Unknown” detection

Logger saves activities in logs/pipeline.log

Handles multiple images per person

Threshold-based recognition to reduce false positives

🔧 Requirements

Python 3.10+
Packages (install via pip install -r requirements.txt)


🖥️ Setup & Usage

1️⃣ Capture Faces

python capture.py --name Alice --count 20 --auto --interval 0.5


--name → Person name (folder will be created under dataset/)

--count → Number of images to capture

--auto → Automatically capture every --interval seconds

--interval → Seconds between captures

2️⃣ Encode Faces

python encode.py

Reads all images in dataset/

Detects faces using InsightFace ONNX model (w600k_r50.onnx)

Saves embeddings and mean embeddings to outputs/encodings/

3️⃣ Train Classifier

python train.py

Loads embeddings from outputs/encodings/

Trains a KNN classifier on the embeddings

Saves trained classifier to outputs/classifier/classifier.pkl

4️⃣ Recognize Faces

python recognize.py

Opens webcam and detects faces in real time

Uses embeddings + trained classifier to recognize known faces

Displays “Unknown” for faces not in the dataset

Shows recognized person name and confidence score


⚠️ Notes

Single-person datasets: Capture one person at a time to ensure correct labeling.

Twins / look-alikes: System may misrecognize very similar faces.

Thresholds: Adjust recognition thresholds in recognize.py to reduce false positives.

Logs: All capture, encoding, training, and recognition activities are logged in logs/pipeline.log.


📦 ONNX Model Used

Model: w600k_r50.onnx (Buffalo-L, ResNet50)

Source: InsightFace on HuggingFace

Purpose: Face detection + 512-dimensional embeddings


🔧 Logger

Logs saved in logs/pipeline.log

Console and file output

UTF-8 encoding for Windows-safe logging


👀 Future Improvements:

Multi-person capture and automatic labeling

Support GPU acceleration for faster encoding

Advanced verification for twins / very similar faces

GUI interface for easier usage


⚡ License

This project is open source and free to use for personal and educational purposes.

