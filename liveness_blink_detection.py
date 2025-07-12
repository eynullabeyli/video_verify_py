import cv2
from deepface import DeepFace
import sys
import os
import contextlib
import time
from datetime import datetime
import uuid
import mimetypes
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# FastAPI app
app = FastAPI()

# Set video source and reference image

video_path = 'assets/yusif.mp4'
reference_image = 'assets/yusiftest.jpg'

# Main logic as a function
def liveness_and_similarity(video_path, reference_image):
    """
    Perform liveness and similarity check on the given video and reference image.
    Returns a dictionary with similarity, distance, verified, threshold, liveness percentage, and timing info.
    """
    # Check if input files exist
    if not os.path.isfile(video_path):
        return {"error": f"Video file not found: {video_path}"}
    if not os.path.isfile(reference_image):
        return {"error": f"Reference image not found: {reference_image}"}

    start_time_dt = datetime.now()
    start_time = time.time()
    start_time_str = start_time_dt.strftime('%Y-%m-%d %H:%M:%S')

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

    # Prepare tmp directory and temp file name with extension from reference image mime type
    mime_type, encoding = mimetypes.guess_type(reference_image)
    if mime_type:
        ext = mimetypes.guess_extension(mime_type) or '.jpg'
    else:
        ext = '.jpg'

    tmp_dir = 'tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    temp_frame_path = os.path.join(tmp_dir, f"{uuid.uuid4()}{ext}")
    live_face_count = 0
    total_sampled = len(frame_indices)

    for idx, frame_num in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
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

    end_time_dt = datetime.now()
    end_time = time.time()
    end_time_str = end_time_dt.strftime('%Y-%m-%d %H:%M:%S')
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
            "liveness": f"{liveness_percentage:.1f}%",
            "start_time": start_time_str,
            "end_time": end_time_str,
            "elapsed_time": f"{elapsed_time:.2f} seconds"
        }
        return result
    else:
        return {"error": "No valid face frames for similarity check."}

# Pydantic model for request
class LivenessAndSimilarityRequest(BaseModel):
    video_path: str
    reference_image: str

# POST endpoint
@app.post("/liveness-and-similarity-check")
async def liveness_and_similarity_check(request: LivenessAndSimilarityRequest):
    result = liveness_and_similarity(request.video_path, request.reference_image)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return JSONResponse(content=result)

# If run as a script, keep CLI functionality
if __name__ == "__main__":
    result = liveness_and_similarity(video_path, reference_image)
    print(result)
