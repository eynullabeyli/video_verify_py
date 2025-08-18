import cv2
from deepface import DeepFace
import os
import time
from datetime import datetime
import uuid
import mimetypes
import ffmpeg
import whisper
from difflib import SequenceMatcher
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import logging

logger = logging.getLogger(__name__)

# Metal GPU support
try:
    import tensorflow as tf
    # tensorflow_metal is automatically loaded when available
    # Check if GPU devices are available (which indicates Metal support on macOS)
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        METAL_AVAILABLE = True
        logger.info("Metal GPU support detected - Found %s GPU device(s)", len(gpu_devices))
    else:
        METAL_AVAILABLE = False
        logger.info("No GPU devices found - Metal GPU not available")
except ImportError:
    METAL_AVAILABLE = False
    logger.info("TensorFlow not available - Metal GPU support not available")

# Global Whisper model cache
_whisper_model = None
_whisper_model_lock = threading.Lock()

def get_whisper_model():
    """
    Get or create Whisper model with caching for better performance.
    """
    global _whisper_model
    with _whisper_model_lock:
        if _whisper_model is None:
            if METAL_GPU_ACTIVE:
                logger.info("Loading Whisper model with Metal GPU acceleration")
            else:
                logger.info("Loading Whisper model on CPU")
            _whisper_model = whisper.load_model("large-v3")
        return _whisper_model

def get_gpu_info():
    """
    Get detailed information about available GPU devices.
    """
    if not METAL_AVAILABLE:
        return "Metal GPU support not available"
    
    try:
        gpu_devices = tf.config.list_physical_devices('GPU')
        cpu_devices = tf.config.list_physical_devices('CPU')
        
        info = f"GPU Devices: {len(gpu_devices)}, CPU Devices: {len(cpu_devices)}\n"
        
        for i, device in enumerate(gpu_devices):
            info += f"  GPU {i}: {device.name}\n"
        
        for i, device in enumerate(cpu_devices):
            info += f"  CPU {i}: {device.name}\n"
        
        return info
    except Exception as e:
        return f"Error getting GPU info: {e}"

def configure_metal_gpu():
    """
    Configure TensorFlow to use Metal GPU if available.
    Returns True if Metal GPU is configured, False otherwise.
    """
    if not METAL_AVAILABLE:
        logger.info("Metal GPU not available - using CPU")
        return False
    
    try:
        # Check if Metal GPU is available
        metal_devices = tf.config.list_physical_devices('GPU')
        if metal_devices:
            logger.info("Found %s Metal GPU device(s):", len(metal_devices))
            for device in metal_devices:
                logger.info("  - %s", device.name)
            
            # Enable memory growth to avoid allocating all GPU memory at once
            for device in metal_devices:
                tf.config.experimental.set_memory_growth(device, True)
                logger.info("Enabled memory growth for %s", device.name)
            
            # Set TensorFlow to use Metal GPU
            tf.config.set_visible_devices(metal_devices, 'GPU')
            logger.info("TensorFlow configured to use Metal GPU")
            return True
        else:
            logger.info("No Metal GPU devices found - using CPU")
            return False
    except Exception as e:
        logger.warning("Error configuring Metal GPU: %s", e)
        logger.info("Falling back to CPU")
        return False

# Configure Metal GPU on module import
METAL_GPU_ACTIVE = configure_metal_gpu()

# Log detailed GPU information
logger.info("GPU Configuration:")
logger.info(get_gpu_info())

def get_negative_words(language_code: str):
    """
    Return a set of negative/undesired words for a given language.
    Supports Azerbaijani (az) and English (en). Falls back to a union.
    """
    az_words = {
        "yox", "deyil", "yalan", "pis", "imtina", "rədd", "rəd", "qadağa",
        "nifrət", "narazı", "qəzəb", "problem", "xeyr", "məqbul", "deyil",
        "mümkün", "deyil", "qeyri-mümkün", "təhlükə", "təhdid", "aldatma", "saxta"
    }
    en_words = {
        "no", "not", "false", "bad", "deny", "reject", "forbidden", "hate",
        "angry", "problem", "danger", "threat", "fraud", "fake", "invalid"
    }
    if language_code and language_code.lower().startswith("az"):
        return az_words
    if language_code and language_code.lower().startswith("en"):
        return en_words
    return az_words | en_words

def detect_negative_words(text: str, language_code: str = "az"):
    """
    Simple lexicon + light regex detection of negative words in the transcribed text.
    Returns a dict with matches, count, total_tokens, and ratio percentage.
    """
    if not text:
        return {"matches": [], "count": 0, "total_tokens": 0, "ratio_pct": 0.0}
    lowered = text.lower()
    cleaned = lowered
    for ch in [".", ",", "!", "?", ";", ":", "(", ")", "[", "]", "{", "}", "\"", "'", "-"]:
        cleaned = cleaned.replace(ch, " ")
    tokens = [t for t in cleaned.split() if t]
    lexicon = get_negative_words(language_code or "az")
    matches = [t for t in tokens if t in lexicon]
    # Light-weight handling of common Azerbaijani negation forms
    if language_code and language_code.lower().startswith("az"):
        patterns = [r"^istəmir", r"^olma", r"^olmaz$", r"^yoxdur$", r"^yoxdu$"]
        regex_hits = [t for t in tokens if any(re.match(p, t) for p in patterns)]
        for t in regex_hits:
            if t not in matches:
                matches.append(t)
    ratio = (len(matches) / len(tokens) * 100.0) if tokens else 0.0
    return {"matches": matches, "count": len(matches), "total_tokens": len(tokens), "ratio_pct": ratio}

def process_frame_parallel(frame_data):
    """
    Process a single frame for face detection and verification.
    Designed to be used with ThreadPoolExecutor for parallel processing.
    """
    frame_num, frame, reference_image_path, temp_frame_path = frame_data
    
    try:
        # Check for face presence (liveness proxy)
        analysis = DeepFace.analyze(frame, actions=["age"], enforce_detection=True)
        
        if isinstance(analysis, list):
            analysis = analysis[0]
        
        # Face detected, proceed to verification
        cv2.imwrite(temp_frame_path, frame)
        
        verification = DeepFace.verify(temp_frame_path, reference_image_path, enforce_detection=False, model_name="ArcFace")
        
        distance = verification.get("distance", None)
        threshold = verification.get("threshold", None)
        
        if distance is not None and threshold is not None:
            try:
                similarity_pct = max(0, (float(threshold) - float(distance)) / float(threshold) * 100)
            except Exception:
                similarity_pct = 0
            
            return {
                'frame_num': frame_num,
                'similarity_pct': similarity_pct,
                'distance': distance,
                'verified': verification.get("verified", False),
                'threshold': threshold,
                'face_detected': True
            }
    except Exception:
        pass
    
    return {
        'frame_num': frame_num,
        'similarity_pct': -1,
        'distance': None,
        'verified': False,
        'threshold': None,
        'face_detected': False
    }

def video_verification(video_bytes, image_bytes, transcribe_reference=None, debug=False):
    """
    Perform video verification: liveness check, face similarity, and transcription on the given video and reference image.
    If transcribe_reference is provided, also compute transcription similarity.
    Returns a dictionary with:
    - similarity: best single-frame similarity to the reference image (image-level)
    - same_person_similarity_percentage: percentage of face-detected frames verified as the same person (video-level)
    - distance, verified, threshold (for the best frame), liveness percentage, timing info, transcription, and transcription similarity.
    If debug is True, prints debug info and includes debug_info in the result.
    """
    if debug:
        logger.debug("Entered video_verification function.")
        logger.debug("video_bytes length: %s", len(video_bytes))
        logger.debug("image_bytes length: %s", len(image_bytes))
        logger.debug("transcribe_reference: %s", transcribe_reference)
        logger.debug("Metal GPU active: %s", METAL_GPU_ACTIVE)
    
    # Log Metal GPU status
    if METAL_GPU_ACTIVE:
        logger.info("Using Metal GPU for video verification")
    else:
        logger.info("Using CPU for video verification")
    
    # Write video_bytes to a temporary file for OpenCV and ffmpeg
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as video_tmp:
        video_tmp.write(video_bytes)
        video_path = video_tmp.name
    if debug:
        logger.debug("Video temp file created at: %s", video_path)
    # Write image_bytes to a temporary file for DeepFace.verify (needs file path)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as img_tmp:
        img_tmp.write(image_bytes)
        reference_image_path = img_tmp.name
    if debug:
        logger.debug("Reference image temp file created at: %s", reference_image_path)

    debug_info = {}
    try:
        start_time_dt = datetime.now()
        start_time = time.time()
        start_time_str = start_time_dt.strftime('%Y-%m-%d %H:%M:%S')
        if debug:
            logger.debug("Start time: %s", start_time_str)
            logger.debug("Opening video with OpenCV...")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if debug:
            logger.debug("OpenCV video opened. Total frames: %s, FPS: %s", total_frames, fps)

        # Calculate frame indices: optimized sampling (one per 2 seconds instead of 1 second)
        if debug:
            logger.debug("Calculating frame indices for sampling...")
        if fps > 0:
            num_seconds = int(total_frames // fps)
            # Sample every 2 seconds instead of every second for better performance
            frame_indices = [int(i * fps) for i in range(0, num_seconds, 2)]
            # Ensure we have at least 3 frames for analysis
            if len(frame_indices) < 3:
                frame_indices = [int(i * fps) for i in range(num_seconds)]
        else:
            frame_indices = list(range(0, total_frames, 2))
        if debug:
            logger.debug("Frame indices to sample: %s", frame_indices)

        # --- Audio extraction and transcription (parallel with video processing) ---
        transcription = None
        detected_language = None
        
        def process_audio():
            nonlocal transcription, detected_language
            try:
                audio_path = video_path + ".mp3"
                if debug:
                    logger.debug("Extracting audio from video to: %s", audio_path)
                (
                    ffmpeg
                    .input(video_path)
                    .output(audio_path, format='mp3', acodec='mp3', ac=2, ar='44100', audio_bitrate='320k')
                    .overwrite_output()
                    .run(quiet=True)
                )
                if debug:
                    logger.debug("Audio extraction complete. Transcribing audio...")
                
                model = get_whisper_model()  # Use cached model
                result = model.transcribe(audio_path, language='az', task='transcribe')
                transcription = result.get("text", None)
                detected_language = result.get("language", None)
                if debug:
                    logger.debug("Transcription result: %s", transcription)
                    logger.debug("Detected language: %s", detected_language)
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    if debug:
                        logger.debug("Audio temp file removed: %s", audio_path)
            except Exception as e:
                transcription = None
                detected_language = None
                if debug:
                    logger.debug("Audio extraction/transcription error: %s", e)
        
        # Start audio processing in parallel
        audio_thread = threading.Thread(target=process_audio)
        audio_thread.start()

        # --- Parallel frame processing ---
        live_face_detected = False
        best_similarity = -1
        best_distance = None
        best_verified = None
        best_idx = -1
        best_threshold = None
        verified_face_count = 0

        tmp_dir = tempfile.gettempdir()
        temp_frame_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.jpg")
        live_face_count = 0
        total_sampled = len(frame_indices)
        if debug:
            logger.debug("Temp frame path for face: %s", temp_frame_path)
            logger.debug("Total sampled frames: %s", total_sampled)

        # Prepare frame data for parallel processing
        frame_data_list = []
        for idx, frame_num in enumerate(frame_indices):
            if debug:
                logger.debug("Sampling frame %s/%s (frame number: %s)...", idx+1, total_sampled, frame_num)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                if debug:
                    logger.debug("Could not read frame %s", frame_num)
                continue
            frame_data_list.append((frame_num, frame, reference_image_path, temp_frame_path))

        cap.release()

        # Process frames sequentially to avoid Metal GPU threading issues
        if debug:
            logger.debug("Starting sequential frame processing...")
        
        for frame_data in frame_data_list:
            result = process_frame_parallel(frame_data)
            if result['face_detected']:
                live_face_detected = True
                live_face_count += 1
                
                if result['similarity_pct'] > best_similarity:
                    if debug:
                        logger.debug("Frame %s is new best match (similarity_pct=%s)", result['frame_num'], result['similarity_pct'])
                    best_similarity = result['similarity_pct']
                    best_distance = result['distance']
                    best_verified = result['verified']
                    best_idx = result['frame_num']
                    best_threshold = result['threshold']
                if result['verified']:
                    verified_face_count += 1

        # Wait for audio processing to complete
        audio_thread.join()

        if os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)
            if debug:
                logger.debug("Temp frame file removed: %s", temp_frame_path)

        # --- Transcription similarity calculation ---
        transcription_similarity = None
        negative_words_info = {"matches": [], "count": 0, "total_tokens": 0, "ratio_pct": 0.0}
        if transcribe_reference and transcription:
            if debug:
                logger.debug("Calculating transcription similarity...")
            transcription_clean = transcription.strip().lower().replace('.', '').replace(',', '')
            reference_clean = transcribe_reference.strip().lower().replace('.', '').replace(',', '')
            matcher = SequenceMatcher(None, transcription_clean, reference_clean)
            transcription_similarity = round(matcher.ratio() * 100, 1)
            if debug:
                logger.debug("Transcription similarity: %s%%", transcription_similarity)
        else:
            if debug:
                logger.debug("Skipping transcription similarity calculation.")
        # Negative words detection regardless of reference text
        if transcription:
            if debug:
                logger.debug("Detecting negative words in transcription...")
            negative_words_info = detect_negative_words(transcription, detected_language or 'az')
            if debug:
                logger.debug("Negative words: %s", negative_words_info)

        end_time_dt = datetime.now()
        end_time = time.time()
        end_time_str = end_time_dt.strftime('%Y-%m-%d %H:%M:%S')
        elapsed_time = end_time - start_time
        if debug:
            logger.debug("End time: %s", end_time_str)
            logger.debug("Elapsed time: %.2f seconds", elapsed_time)

        if total_sampled > 0:
            liveness_percentage = (live_face_count / total_sampled) * 100
        else:
            liveness_percentage = 0
        if debug:
            logger.debug("Liveness percentage: %s%% (%s/%s)", liveness_percentage, live_face_count, total_sampled)

        if best_similarity >= 0:
            if debug:
                logger.debug("Returning successful result.")
            if live_face_count > 0:
                same_person_similarity_percentage = round((verified_face_count / live_face_count) * 100, 1)
            else:
                same_person_similarity_percentage = None
            result = {
                "similarity": f"{best_similarity:.1f}%",
                "same_person_similarity_percentage": f"{same_person_similarity_percentage}%" if same_person_similarity_percentage is not None else None,
                "distance": best_distance,
                "verified": best_verified,
                "threshold": best_threshold,
                "liveness": f"{liveness_percentage:.1f}%",
                "start_time": start_time_str,
                "end_time": end_time_str,
                "elapsed_time": f"{elapsed_time:.2f} seconds",
                "transcription": transcription,
                "transcription_language": detected_language,
                "transcription_similarity": f"{transcription_similarity}%" if transcription_similarity is not None else None,
                "contains_negative_words": negative_words_info["count"] > 0,
                "negative_words_detected": negative_words_info["matches"],
                "negative_word_count": negative_words_info["count"],
                "negative_word_ratio": f"{negative_words_info['ratio_pct']:.1f}%"
            }
            if debug:
                debug_info.update({
                    "video_path": video_path,
                    "reference_image_path": reference_image_path,
                    "frame_indices": frame_indices,
                    "best_idx": best_idx,
                    "live_face_count": live_face_count,
                    "verified_face_count": verified_face_count,
                    "total_sampled": total_sampled,
                    "liveness_percentage": liveness_percentage,
                    "transcription": transcription,
                    "transcription_similarity": transcription_similarity,
                    "negative_words_info": negative_words_info
                })
                result["debug_info"] = debug_info
            return result
        else:
            if debug:
                logger.debug("No valid face frames for similarity check. Returning error result.")
            result = {
                "error": "No valid face frames for similarity check.",
                "same_person_similarity_percentage": None,
                "transcription": transcription,
                "transcription_language": detected_language,
                "transcription_similarity": f"{transcription_similarity}%" if transcription_similarity is not None else None,
                "contains_negative_words": (negative_words_info["count"] > 0) if transcription else None,
                "negative_words_detected": negative_words_info["matches"] if transcription else None,
                "negative_word_count": negative_words_info["count"] if transcription else None,
                "negative_word_ratio": (f"{negative_words_info['ratio_pct']:.1f}%" if transcription else None)
            }
            if debug:
                debug_info.update({
                    "video_path": video_path,
                    "reference_image_path": reference_image_path,
                    "frame_indices": frame_indices,
                    "live_face_count": live_face_count,
                    "verified_face_count": verified_face_count,
                    "total_sampled": total_sampled,
                    "liveness_percentage": liveness_percentage,
                    "transcription": transcription,
                    "transcription_similarity": transcription_similarity,
                    "negative_words_info": negative_words_info
                })
                result["debug_info"] = debug_info
            return result
    finally:
        if debug:
            logger.debug("Cleaning up temp files...")
        # Clean up temp files
        if os.path.exists(video_path):
            os.remove(video_path)
            if debug:
                logger.debug("Video temp file removed: %s", video_path)
        if os.path.exists(reference_image_path):
            os.remove(reference_image_path)
            if debug:
                logger.debug("Reference image temp file removed: %s", reference_image_path)
        if debug:
            logger.debug("Exiting video_verification function.")