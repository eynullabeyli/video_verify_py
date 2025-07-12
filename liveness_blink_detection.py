import cv2
import dlib
import time
from imutils import face_utils
import numpy as np

# Eye aspect ratio calculation
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])  # vertical distance 1
    B = np.linalg.norm(eye[2] - eye[4])  # vertical distance 2
    C = np.linalg.norm(eye[0] - eye[3])  # horizontal distance
    ear = (A + B) / (2.0 * C)
    return ear

# Constants for blink detection
EYE_AR_THRESH = 0.25  # below this means eye closed
EYE_AR_CONSEC_FRAMES = 3  # consecutive frames to count a blink

# Initialize blink counters
COUNTER = 0
TOTAL = 0

# Initialize head turn and mouth open counters
HEAD_TURN_TOTAL = 0
MOUTH_OPEN_TOTAL = 0

# Initialize eye area variables for occlusion detection
leftEye_area = 0.0
rightEye_area = 0.0

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Get indexes for left and right eyes from facial landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# Get indexes for mouth and nose
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

# For head turn detection, store initial nose x position
initial_nose_x = None
HEAD_TURN_THRESH = 20  # pixels, adjust as needed
MOUTH_OPEN_THRESH = 18  # vertical distance, adjust as needed

# Open video file or webcam
video_path = "video.mp4"  # or 0 for webcam
cap = cv2.VideoCapture(video_path)

total_frames = 0  # Add this line to count total frames

start_time = time.time()
live_threshold = 3  # require at least 3 blinks to consider live

while True:
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1  # Increment frame count

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        nose = shape[nStart:nEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Calculate eye areas for occlusion detection
        leftEye_area = float(cv2.contourArea(leftEye))
        rightEye_area = float(cv2.contourArea(rightEye))
        EYE_AREA_THRESH = 10  # Minimum area for an eye to be considered visible (adjust as needed)
        EYE_AREA_RATIO_THRESH = 0.5  # Minimum ratio between left and right eye area (for symmetry)
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        EYE_AR_DIFF_THRESH = 0.15  # Maximum allowed difference between left and right EAR

        # Calculate mouth area and aspect ratio for plausibility
        mouth_area = float(cv2.contourArea(mouth))
        mouth_width = np.linalg.norm(mouth[0] - mouth[6])
        mouth_height = np.linalg.norm(mouth[3] - mouth[9])
        mouth_aspect_ratio = mouth_height / (mouth_width + 1e-6)
        MOUTH_AREA_THRESH = 20  # Minimum area for mouth to be considered visible
        MOUTH_AR_MIN = 0.2  # Minimum plausible aspect ratio for mouth
        MOUTH_AR_MAX = 1.0  # Maximum plausible aspect ratio for mouth

        # Plausibility checks
        eyes_plausible = (
            leftEye_area > EYE_AREA_THRESH and
            rightEye_area > EYE_AREA_THRESH and
            min(leftEye_area, rightEye_area) / max(leftEye_area, rightEye_area) > EYE_AREA_RATIO_THRESH and
            abs(leftEAR - rightEAR) < EYE_AR_DIFF_THRESH
        )
        mouth_plausible = (
            mouth_area > MOUTH_AREA_THRESH and
            MOUTH_AR_MIN < mouth_aspect_ratio < MOUTH_AR_MAX
        )

        # Draw eye contours
        cv2.polylines(frame, [cv2.convexHull(leftEye)], True, (0,255,0), 1)
        cv2.polylines(frame, [cv2.convexHull(rightEye)], True, (0,255,0), 1)
        # Draw mouth and nose contours
        cv2.polylines(frame, [cv2.convexHull(mouth)], True, (255,0,0), 1)
        cv2.polylines(frame, [cv2.convexHull(nose)], True, (0,0,255), 1)

        # --- Blink Detection (existing) ---
        if eyes_plausible:
            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                    print(f"Blinked! Total blinks: {TOTAL}")
                COUNTER = 0

        # --- Head Turn Detection ---
        if eyes_plausible:
            nose_x = np.mean(nose[:, 0])
            if initial_nose_x is None:
                initial_nose_x = nose_x
            else:
                if abs(nose_x - initial_nose_x) > HEAD_TURN_THRESH:
                    HEAD_TURN_TOTAL += 1
                    print(f"Head turned! Total head turns: {HEAD_TURN_TOTAL}")
                    initial_nose_x = nose_x  # reset reference to avoid multiple counts for same turn

        # --- Mouth Open Detection ---
        if mouth_plausible:
            # Use vertical distance between upper and lower lip
            top_lip = mouth[3]
            bottom_lip = mouth[9]
            mouth_open_dist = np.linalg.norm(top_lip - bottom_lip)
            if mouth_open_dist > MOUTH_OPEN_THRESH:
                MOUTH_OPEN_TOTAL += 1
                print(f"Mouth opened! Total mouth opens: {MOUTH_OPEN_TOTAL}")
                # Wait until mouth closes to count again
                while True:
                    ret2, frame2 = cap.read()
                    if not ret2:
                        break
                    total_frames += 1
                    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                    rects2 = detector(gray2, 0)
                    if len(rects2) == 0:
                        break
                    shape2 = predictor(gray2, rects2[0])
                    shape2 = face_utils.shape_to_np(shape2)
                    mouth2 = shape2[mStart:mEnd]
                    mouth_area2 = float(cv2.contourArea(mouth2))
                    mouth_width2 = np.linalg.norm(mouth2[0] - mouth2[6])
                    mouth_height2 = np.linalg.norm(mouth2[3] - mouth2[9])
                    mouth_aspect_ratio2 = mouth_height2 / (mouth_width2 + 1e-6)
                    mouth_plausible2 = (
                        mouth_area2 > MOUTH_AREA_THRESH and
                        MOUTH_AR_MIN < mouth_aspect_ratio2 < MOUTH_AR_MAX
                    )
                    top_lip2 = mouth2[3]
                    bottom_lip2 = mouth2[9]
                    mouth_open_dist2 = np.linalg.norm(top_lip2 - bottom_lip2)
                    if mouth_open_dist2 < MOUTH_OPEN_THRESH or not mouth_plausible2:
                        break

        cv2.putText(frame, f"Blinks: {TOTAL}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, f"Head Turns: {HEAD_TURN_TOTAL}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        cv2.putText(frame, f"Mouth Opens: {MOUTH_OPEN_TOTAL}", (10,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Liveness Blink Detection", frame)

    # If enough blinks detected, assume live
    if TOTAL >= live_threshold:
        # Only confirm liveness if both eyes are open, both eyes are visible, and face is fully visible
        if (
            len(rects) > 0 and
            len(shape) == 68 and
            leftEAR > EYE_AR_THRESH and
            rightEAR > EYE_AR_THRESH and
            leftEye_area > EYE_AREA_THRESH and
            rightEye_area > EYE_AREA_THRESH and
            eyes_plausible and
            mouth_plausible
        ):
            print("Liveness confirmed (enough blinks detected, both eyes open, both eyes visible, face fully visible, and plausible facial features)")
            break
        else:
            print("Liveness not confirmed: face not fully visible, covered, one/both eyes are closed, one/both eyes are not visible, or facial features are implausible.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Add liveness percentage calculation and print
if total_frames > 0:
    # Calculate combined liveness events
    total_events = TOTAL + HEAD_TURN_TOTAL + MOUTH_OPEN_TOTAL
    liveness_percentage = (total_events / total_frames) * 100
    remaining_liveness = 100 - liveness_percentage
    print(f"Liveness Remaining: {remaining_liveness:.2f}%")
    print(f"Total Blinks: {TOTAL}")
    print(f"Total Head Turns: {HEAD_TURN_TOTAL}")
    print(f"Total Mouth Opens: {MOUTH_OPEN_TOTAL}")
else:
    print("No frames processed.")
