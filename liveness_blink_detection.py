import cv2
from deepface import DeepFace

# Set video source: 0 for webcam or provide a video file path
video_path = 0  # or 'video.mp4'
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Analyze frame for liveness (anti-spoofing)
    try:
        result = DeepFace.analyze(frame, actions=["anti_spoof"], enforce_detection=False)
        liveness = result.get("anti_spoof", "unknown")
    except Exception as e:
        liveness = f"error: {e}"

    # Show liveness result on frame
    cv2.putText(frame, f"Liveness: {liveness}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if liveness == "real" else (0, 0, 255), 2)
    print(f"Liveness: {liveness}")

    cv2.imshow("Liveness Detection (DeepFace)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
