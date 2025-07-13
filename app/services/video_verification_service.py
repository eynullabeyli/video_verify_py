import cv2
from deepface import DeepFace
import os
import contextlib
import time
from datetime import datetime
import uuid
import mimetypes
import ffmpeg
import whisper
from difflib import SequenceMatcher
import tempfile

def video_verification(video_bytes, image_bytes, transcribe_reference=None):
    """
    Perform video verification: liveness check, face similarity, and transcription on the given video and reference image.
    If transcribe_reference is provided, also compute transcription similarity.
    Returns a dictionary with similarity, distance, verified, threshold, liveness percentage, timing info, transcription, and transcription similarity.
    """
    # Write video_bytes to a temporary file for OpenCV and ffmpeg
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as video_tmp:
        video_tmp.write(video_bytes)
        video_path = video_tmp.name
    # Write image_bytes to a temporary file for DeepFace.verify (needs file path)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as img_tmp:
        img_tmp.write(image_bytes)
        reference_image_path = img_tmp.name

    try:
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

        live_face_detected = False
        best_similarity = -1
        best_distance = None
        best_verified = None
        best_idx = -1
        best_threshold = None

        tmp_dir = tempfile.gettempdir()
        temp_frame_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.jpg")
        live_face_count = 0
        total_sampled = len(frame_indices)

        # --- Audio extraction and transcription ---
        transcription = None
        detected_language = None
        try:
            audio_path = video_path + ".mp3"
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, format='mp3', acodec='mp3', ac=2, ar='44100', audio_bitrate='320k')
                .overwrite_output()
                .run(quiet=True)
            )
            model = whisper.load_model("medium")
            result = model.transcribe(audio_path, language='az', task='transcribe')
            transcription = result.get("text", None)
            detected_language = result.get("language", None)
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            transcription = None
            detected_language = None
        # --- End audio extraction and transcription ---

        transcription_similarity = None
        if transcribe_reference and transcription:
            # Remove periods (dots) and commas from both strings before comparison
            transcription_clean = transcription.strip().lower().replace('.', '').replace(',', '')
            reference_clean = transcribe_reference.strip().lower().replace('.', '').replace(',', '')
            matcher = SequenceMatcher(None, transcription_clean, reference_clean)
            transcription_similarity = round(matcher.ratio() * 100, 1)

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
                        verification = DeepFace.verify(temp_frame_path, reference_image_path, enforce_detection=False, model_name="ArcFace")
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
                "elapsed_time": f"{elapsed_time:.2f} seconds",
                "transcription": transcription,
                "transcription_language": detected_language,
                "transcription_similarity": f"{transcription_similarity}%" if transcription_similarity is not None else None
            }
            return result
        else:
            return {"error": "No valid face frames for similarity check.", "transcription": transcription, "transcription_language": detected_language, "transcription_similarity": f"{transcription_similarity}%" if transcription_similarity is not None else None}
    finally:
        # Clean up temp files
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(reference_image_path):
            os.remove(reference_image_path) 