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
max_frames = 11  # Number of frames to sample evenly across the video

start_time = time.time()
start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Get total number of frames in the video
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

# Calculate frame indices to sample
if max_frames > total_frames:
    frame_indices = list(range(total_frames))
else:
    frame_indices = [int(i * total_frames / max_frames) for i in range(max_frames)]

temp_frame_path = "_temp_face_for_similarity.jpg"
live_face_detected = False
best_similarity = -1
best_distance = None
best_verified = None
best_idx = -1
best_threshold = None

for idx, frame_num in enumerate(frame_indices):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        continue
    # Print progress every 2 frames
    if (idx + 1) % 2 == 0:
        print(f"Processed {idx + 1} of {len(frame_indices)} sampled frames...")
    # Check for face presence (liveness proxy)
    try:
        with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            analysis = DeepFace.analyze(frame, actions=["age"], enforce_detection=True)
        if isinstance(analysis, list):
            analysis = analysis[0]
        # If a face is detected, proceed to similarity check
        live_face_detected = True
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
                    best_idx = idx
                    best_threshold = threshold
        except Exception as e:
            continue
    except Exception as e:
        continue
if os.path.exists(temp_frame_path):
    os.remove(temp_frame_path)

end_time = time.time()
end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
elapsed_time = end_time - start_time

if live_face_detected:
    print("Live face detected in at least one frame.")
else:
    print("No live face detected in any sampled frame.")
if best_similarity >= 0:
    print(f"[ArcFace] Highest similarity: {best_similarity:.1f}%, distance={best_distance}, verified={best_verified}, threshold={best_threshold}")
else:
    print("No valid face frames for similarity check.")
print(f"Start time: {start_time_str}")
print(f"End time:   {end_time_str}")
print(f"Total elapsed time: {elapsed_time:.2f} seconds")
