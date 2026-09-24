import os
import sys
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
# pyrefly: ignore [missing-import]
from deep_sort_realtime.deepsort_tracker import DeepSort

# Avoid OpenMP conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ----------------- PATH CONFIGURATION -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

GUN_WEIGHTS = os.path.join(PROJECT_ROOT, "EL", "final", "guns", "weights", "best.pt")
HELMET_WEIGHTS = os.path.join(PROJECT_ROOT, "EL", "final", "helmets", "weights", "best.pt")
MASK_WEIGHTS = os.path.join(PROJECT_ROOT, "EL", "final", "ski mask", "weights", "best.pt")
RUN_WEIGHTS = os.path.join(PROJECT_ROOT, "EL", "final", "yolo11n.pt")
KNIFE_WEIGHTS = os.path.join(PROJECT_ROOT, "runs", "detect", "train", "weights", "best.pt")

print(" Loading AI Models...")
model_gun = YOLO(GUN_WEIGHTS)
model_helmet = YOLO(HELMET_WEIGHTS)
model_mask = YOLO(MASK_WEIGHTS)
model_run = YOLO(RUN_WEIGHTS)

# Optional knife model if trained weights exist
model_knife = YOLO(KNIFE_WEIGHTS) if os.path.exists(KNIFE_WEIGHTS) else None

# Deep SORT tracker for human movement & speed estimation
tracker = DeepSort(max_age=30)

# ----------------- VIDEO INPUT CONFIGURATION -----------------
# Priority: 1) Command line arg -> 2) Sample test video -> 3) Webcam (0)
default_video = os.path.join(PROJECT_ROOT, "running output videos", "test vid 1.mp4")

if len(sys.argv) > 1:
    video_source = sys.argv[1]
    if video_source.isdigit():
        video_source = int(video_source)
elif os.path.exists(default_video):
    video_source = default_video
else:
    video_source = 0

print(f"📹 Opening video source: {video_source}")
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"❌ Error: Unable to open video source '{video_source}'")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer
output_video_path = os.path.join(BASE_DIR, "output_combined.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Directory for suspicious detections
output_dir = os.path.join(BASE_DIR, "suspicious_frames")
os.makedirs(output_dir, exist_ok=True)

# ----------------- STATE VARIABLES -----------------
frame_idx = 0
frame_skip = 5
running_frames = 0
run_threshold = 10
prev_positions = {}

# Summary tracking
detected_events = []
highest_rating = 1

print("\n🚀 Starting Multi-Threat Surveillance Detection...")
print("Press 'q' in the display window to exit anytime.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    suspicious = False
    rating = 1
    alert_labels = []

    # Run heavy object detections every N frames for performance
    if frame_idx % frame_skip == 0:
        # 1. GUN DETECTION (5-Star Threat)
        results_gun = model_gun(frame, verbose=False)[0]
        if results_gun.boxes is not None and len(results_gun.boxes) > 0:
            rating = max(rating, 5)
            alert_labels.append("Gun")
            suspicious = True
            for box in results_gun.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"GUN {conf:.2f}", (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 2. KNIFE DETECTION (5-Star Threat, if model available)
        if model_knife:
            results_knife = model_knife(frame, verbose=False)[0]
            if results_knife.boxes is not None and len(results_knife.boxes) > 0:
                rating = max(rating, 5)
                alert_labels.append("Knife")
                suspicious = True
                for box in results_knife.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"KNIFE {conf:.2f}", (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 3. MASK DETECTION (4-Star Threat: class 0 = face cover / ski mask)
        results_mask = model_mask(frame, verbose=False)[0]
        if results_mask.boxes is not None:
            for box in results_mask.boxes:
                if int(box.cls[0]) == 0:
                    rating = max(rating, 4)
                    if "Mask" not in alert_labels:
                        alert_labels.append("Mask")
                    suspicious = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
                    cv2.putText(frame, f"MASK {conf:.2f}", (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

        # 4. HELMET DETECTION (4-Star Threat: class 0 = helmet)
        results_helmet = model_helmet(frame, verbose=False)[0]
        if results_helmet.boxes is not None:
            for box in results_helmet.boxes:
                if int(box.cls[0]) == 0:
                    rating = max(rating, 4)
                    if "Helmet" not in alert_labels:
                        alert_labels.append("Helmet")
                    suspicious = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                    cv2.putText(frame, f"HELMET {conf:.2f}", (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # 5. HUMAN MOVEMENT & SPEED ESTIMATION (DeepSORT)
    results_person = model_run(frame, classes=[0], verbose=False)[0]
    detections = []
    if results_person.boxes is not None:
        for box in results_person.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

    tracks = tracker.update_tracks(detections, frame=frame)
    someone_running = False

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        cx, cy = (l + r) // 2, (t + b) // 2

        prev_positions.setdefault(track_id, []).append((frame_idx, cx, cy))
        prev_positions[track_id] = prev_positions[track_id][-10:]

        speed = 0.0
        if len(prev_positions[track_id]) >= 2:
            f1, x1, y1 = prev_positions[track_id][0]
            f2, x2, y2 = prev_positions[track_id][-1]
            dt = (f2 - f1) / fps
            if dt > 0:
                dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                speed = dist / dt

        if speed > 55:
            running_frames += 1
            someone_running = True
            color = (0, 0, 255)
            status = "Running"
        else:
            color = (0, 255, 0)
            status = "Walking"

        cv2.rectangle(frame, (l, t), (r, b), color, 2)
        cv2.putText(frame, f"ID {track_id} {status}", (l, max(20, t - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if not someone_running:
        running_frames = 0

    # Running ratings (3-Star for prolonged, 2-Star for short)
    if running_frames >= 2 * run_threshold:
        rating = max(rating, 3)
        if "Prolonged Running" not in alert_labels:
            alert_labels.append("Prolonged Running")
        suspicious = True
    elif running_frames >= run_threshold:
        rating = max(rating, 2)
        if "Short Running" not in alert_labels:
            alert_labels.append("Short Running")
        suspicious = True

    highest_rating = max(highest_rating, rating)

    # Visual HUD banner at the top of the frame
    status_text = f"Threat: {rating}-Star" + (f" [{', '.join(alert_labels)}]" if alert_labels else " [Normal]")
    banner_color = (0, 0, 255) if rating >= 4 else ((0, 165, 255) if rating >= 2 else (0, 200, 0))
    cv2.rectangle(frame, (0, 0), (width, 36), (20, 20, 20), -1)
    cv2.putText(frame, status_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner_color, 2)
    cv2.putText(frame, f"Time: {timestamp:.2f}s", (width - 160, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Save suspicious snapshot frame
    if suspicious:
        filename = os.path.join(output_dir, f"frame_{frame_idx}_rating{rating}.jpg")
        cv2.imwrite(filename, frame)
        detected_events.append((frame_idx, timestamp, rating, list(alert_labels)))
        print(f"⭐ [Frame {frame_idx:04d} | {timestamp:6.2f}s] {rating}-Star Threat: {', '.join(alert_labels)}")

    out.write(frame)
    cv2.imshow("Multi-Threat Crime Detection & Surveillance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n⏹ Processing interrupted by user.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# ----------------- SUMMARY REPORT -----------------
print("\n" + "=" * 50)
print("📊 SURVEILLANCE & THREAT DETECTION SUMMARY")
print("=" * 50)
print(f"Total Frames Processed: {frame_idx}")
print(f"Highest Threat Rating:  {highest_rating} Star(s)")
print(f"Suspicious Events Logged: {len(detected_events)}")
print(f"Annotated Video Saved:  {output_video_path}")
print(f"Detection Frames Saved: {output_dir}/")
print("=" * 50)
