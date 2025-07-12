import cv2
from deepface import DeepFace
from collections import Counter
import sys
import os
import contextlib
import time

# Set video source: 0 for webcam or provide a video file path
video_path = 'yusif.mp4'  # Use video file instead of webcam
cap = cv2.VideoCapture(video_path)

emotions = []
ages = []
genders = []
races = []
first_face_frame = None

frame_count = 11
frame_skip = 5  # Analyze every 10th frame for much faster processing

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        # Show frame without analysis
        cv2.imshow("DeepFace Attribute Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

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
        # Compose a short summary
        summary = f"{emotion}, {age}, {dominant_gender}, {race}"
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
        summary = f"error: {e}"

    # Show only the last result on the frame
    cv2.putText(frame, summary, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # print(summary)  # Remove this line to avoid printing every frame

    cv2.imshow("DeepFace Attribute Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Calculate and print elapsed time
elapsed_time = time.time() - start_time

# Print summary of all results
if emotions:
    most_common_emotion = Counter(emotions).most_common(1)[0][0]
else:
    most_common_emotion = 'unknown'
if ages:
    avg_age = round(sum(ages) / len(ages), 1)
else:
    avg_age = 'unknown'
if genders:
    most_common_gender = Counter(genders).most_common(1)[0][0]
else:
    most_common_gender = 'unknown'
if races:
    most_common_race = Counter(races).most_common(1)[0][0]
else:
    most_common_race = 'unknown'

print(f"Summary: {most_common_emotion}, {avg_age}, {most_common_gender}, {most_common_race}")
print(f"Total elapsed time: {elapsed_time:.2f} seconds")

# Similarity check with yusif.jpeg using only ArcFace
if first_face_frame is not None:
    temp_frame_path = "_temp_first_face.jpeg"
    cv2.imwrite(temp_frame_path, first_face_frame)
    try:
        with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            verification = DeepFace.verify(temp_frame_path, "yusif.jpeg", enforce_detection=False, model_name="ArcFace")
        verified = verification.get("verified", False)
        distance = verification.get("distance", None)
        model_name = verification.get("model", "ArcFace")
        threshold = verification.get("threshold", None)
        if distance is not None and threshold is not None:
            try:
                similarity_pct = max(0, (float(threshold) - float(distance)) / float(threshold) * 100)
            except Exception:
                similarity_pct = 0
            print(f"[{model_name}] verified={verified}, distance={distance}, similarity={similarity_pct:.1f}%, threshold={threshold}")
        else:
            print(f"[{model_name}] verified={verified}, distance={distance}, threshold={threshold}")
    except Exception as e:
        print(f"[ArcFace] Similarity check failed: {e}")
    os.remove(temp_frame_path)
else:
    print("No face detected in video for similarity check.")
