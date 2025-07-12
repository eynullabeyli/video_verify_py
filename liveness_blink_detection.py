import cv2
from deepface import DeepFace
from collections import Counter
import sys
import os
import contextlib
import time
from datetime import datetime

# Set video source: 0 for webcam or provide a video file path
video_path = 'yusif.mp4'  # Use video file instead of webcam
cap = cv2.VideoCapture(video_path)

emotions = []
ages = []
genders = []
races = []
first_face_frame = None

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
    # Analyze frame for available DeepFace actions
    try:
        with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            result = DeepFace.analyze(frame, actions=["emotion", "age", "gender", "race"], enforce_detection=False)
        if isinstance(result, list):
            result = result[0]
        emotion = result.get("dominant_emotion", "unknown")
        age = result.get("age", None)
        gender = result.get("gender", "unknown")
        race = result.get("dominant_race", "unknown")
        # Determine dominant gender
        if isinstance(gender, dict):
            dominant_gender = max(gender, key=lambda k: gender[k])
        else:
            dominant_gender = gender
        # Collect for summary
        emotions.append(emotion)
        if age is not None and isinstance(age, (int, float)):
            ages.append(age)
        genders.append(dominant_gender)
        races.append(race)
        # Save the first frame with a detected face for similarity check
        if first_face_frame is None and emotion != "unknown":
            first_face_frame = frame.copy()
    except Exception as e:
        continue

cap.release()
cv2.destroyAllWindows()

# Compare all detected faces in the video to yusifnew.jpg using ArcFace, print the highest similarity found
if emotions:
    temp_frame_path = "_temp_face_for_similarity.jpg"
    best_similarity = -1
    best_distance = None
    best_verified = None
    best_idx = -1
    best_threshold = None
    for idx, (emotion, age, gender, race) in enumerate(zip(emotions, ages + [None]*(len(emotions)-len(ages)), genders, races)):
        # Only compare if a face was detected (emotion != 'unknown')
        if emotion == 'unknown':
            continue
        # Get the corresponding sampled frame
        frame_num = frame_indices[idx]
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            continue
        cv2.imwrite(temp_frame_path, frame)
        try:
            with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                verification = DeepFace.verify(temp_frame_path, "yusifnew.jpg", enforce_detection=False, model_name="ArcFace")
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
    if os.path.exists(temp_frame_path):
        os.remove(temp_frame_path)
    if best_similarity >= 0:
        print(f"[ArcFace] Highest similarity: {best_similarity:.1f}%, distance={best_distance}, verified={best_verified}, threshold={best_threshold}")
    else:
        print("No valid face frames for similarity check.")
    end_time = time.time()
    end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    elapsed_time = end_time - start_time
    print(f"Start time: {start_time_str}")
    print(f"End time:   {end_time_str}")
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")
else:
    print("No face detected in video for similarity check.")
