from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from app.services.video_verification_service import video_verification
import shutil
import os
import uuid
import mimetypes
from typing import Optional

app = FastAPI(
    title="Video Verification API",
    description="API for video verification: liveness check, face similarity, and video-to-text transcription using video and reference image uploads.",
    version="1.0.0"
)

@app.post(
    "/video-verification-check",
    summary="Video Verification (Liveness, Similarity, Transcription)",
    description="Upload a video file and a reference image. Optionally, provide reference text for transcription similarity check. The service will check for liveness, face similarity, transcribe the video, and compare transcription to the reference text.",
    tags=["Video Verification"]
)
async def video_verification_check(
    video_file: UploadFile = File(
        ..., 
        description="Video file (must be a video type, e.g., mp4, avi)",
        openapi_extra={"accept": "video/*"}
    ),
    reference_image: UploadFile = File(
        ..., 
        description="Reference image (must be an image type, e.g., jpg, png)",
        openapi_extra={"accept": "image/*"}
    ),
    transcribe_reference: Optional[str] = File(
        default="Mən Nicat Soltanov kredit almaq istəyirəm",
        description="Optional reference text for transcription similarity check."
    )
):
    # Validate file types
    video_mime = video_file.content_type
    image_mime = reference_image.content_type
    if not video_mime or not video_mime.startswith("video/"):
        raise HTTPException(status_code=400, detail="video_file must be a video type.")
    if not image_mime or not image_mime.startswith("image/"):
        raise HTTPException(status_code=400, detail="reference_image must be an image type.")

    # Save uploaded files to tmp directory
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    video_path = os.path.join(tmp_dir, f"{uuid.uuid4()}_{video_file.filename}")
    ref_path = os.path.join(tmp_dir, f"{uuid.uuid4()}_{reference_image.filename}")

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video_file.file, f)
    with open(ref_path, "wb") as f:
        shutil.copyfileobj(reference_image.file, f)

    # Call the service for video verification
    result = video_verification(video_path, ref_path, transcribe_reference)

    # Clean up uploaded files
    os.remove(video_path)
    os.remove(ref_path)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return JSONResponse(content=result) 