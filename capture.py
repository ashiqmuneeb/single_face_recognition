import cv2, os, time, argparse
from logger import get_logger

logger = get_logger("capture")

def capture_person(name, out_dir="dataset", count=20, auto=False, interval=0.5):
    person_dir = os.path.join(out_dir, name)
    os.makedirs(person_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam.")
        return
    logger.info("Webcam opened. Press 'c' to capture manually, 'q' to quit.")

    captured = 0
    last_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Could not read from webcam.")
            break

        h, w = frame.shape[:2]
        cv2.putText(frame, f"{name} — captured {captured}/{count if count else '∞'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if auto:
            if captured < count and (time.time() - last_time) >= interval:
                fname = os.path.join(person_dir, f"{int(time.time()*1000)}.jpg")
                cv2.imwrite(fname, frame)
                logger.info(f"Saved (auto): {fname}")
                captured += 1
                last_time = time.time()
                if captured >= count:
                    break
        else:
            if key == ord('c'):
                fname = os.path.join(person_dir, f"{int(time.time()*1000)}.jpg")
                cv2.imwrite(fname, frame)
                logger.info(f"Saved (manual): {fname}")
                captured += 1
                if count and captured >= count:
                    break

    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"Done. Images saved to: {person_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Person name (folder will be created)")
    parser.add_argument("--out", default="dataset", help="Output dataset directory")
    parser.add_argument("--count", type=int, default=20, help="How many images to capture")
    parser.add_argument("--auto", action="store_true", help="Auto-capture every --interval seconds")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between auto captures")
    args = parser.parse_args()
    capture_person(args.name, args.out, args.count, args.auto, args.interval)
