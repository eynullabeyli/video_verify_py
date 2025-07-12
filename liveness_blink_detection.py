import cv2
from deepface import DeepFace
import sys
import os
import contextlib
import time
from datetime import datetime

# Set video source and reference image
video_path = 'yusif.mp4'
reference_image = 'yusifnew.jpg'

start_time = time.time()
start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate frame indices: one per second
if fps > 0:
    num_seconds = int(total_frames // fps)
    frame_indices = [int(i * fps) for i in range(num_seconds)]
else:
    frame_indices = list(range(total_frames))

# Prepare for liveness and similarity checks
live_face_detected = False
best_similarity = -1
best_distance = None
best_verified = None
best_idx = -1
best_threshold = None
temp_frame_path = "_temp_face_for_similarity.jpg"
live_face_count = 0
total_sampled = len(frame_indices)

for idx, frame_num in enumerate(frame_indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        continue
    # Print progress every 5 frames
    if (idx + 1) % 5 == 0:
        print(f"Processed {idx + 1} of {total_sampled} sampled frames...")
    # Check for face presence (liveness proxy)
    try:
        with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            analysis = DeepFace.analyze(frame, actions=["age"], enforce_detection=True)
        if isinstance(analysis, list):
            analysis = analysis[0]
        # If a face is detected, proceed to similarity check
        live_face_detected = True
        live_face_count += 1
        cv2.imwrite(temp_frame_path, frame)
        try:
            with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                verification = DeepFace.verify(temp_frame_path, reference_image, enforce_detection=False, model_name="ArcFace")
            distance = verification.get("distance", None)
            threshold = verification.get("threshold", None)
            if distance is not None and threshold is not None:
                try:
                    similarity_pct = max(0, (float(threshold) - float(distance)) / float(threshold) * 100)
                except Exception:
                    similarity_pct = 0
                if similarity_pct > best_similarity:
                    best_similarity = similarity_pct
                    best_distance = distance
                    best_verified = verification.get("verified", False)
                    best_idx = frame_num
                    best_threshold = threshold
        except Exception as e:
            continue
    except Exception as e:
        continue
cap.release()
if os.path.exists(temp_frame_path):
    os.remove(temp_frame_path)

end_time = time.time()
end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
elapsed_time = end_time - start_time

if total_sampled > 0:
    liveness_percentage = (live_face_count / total_sampled) * 100
else:
    liveness_percentage = 0

if best_similarity >= 0:
    result = {
        "similarity": f"{best_similarity:.1f}%",
        "distance": best_distance,
        "verified": best_verified,
        "threshold": best_threshold,
        "liveness": f"{liveness_percentage:.1f}%"
    }
    print(result)
else:
    print("No valid face frames for similarity check.")
print(f"Start time: {start_time_str}")
print(f"End time:   {end_time_str}")
print(f"Total elapsed time: {elapsed_time:.2f} seconds")
