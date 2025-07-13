from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from app.services.video_verification_service import video_verification
import shutil
import os
import uuid
import mimetypes
from typing import Optional
import logging
import traceback
from io import BytesIO

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Video Verification API",
    description="API for video verification: liveness check, face similarity, and video-to-text transcription using video and reference image uploads.",
    version="1.0.0"
)

@app.exception_handler(Exception)
def global_exception_handler(request, exc):
    logging.error(f"Unhandled error: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )

@app.post(
    "/video-verification-check",
    summary="Video Verification (Liveness, Similarity, Transcription)",
    description="Upload a video file and a reference image. Optionally, provide reference text for transcription similarity check. The service will check for liveness, face similarity, transcribe the video, and compare transcription to the reference text.",
    tags=["Video Verification"]
)
def video_verification_check(
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
    logging.info("[API] Received /video-verification-check request")
    # Validate file types
    video_mime = video_file.content_type
    image_mime = reference_image.content_type
    if not video_mime or not video_mime.startswith("video/"):
        logging.error("[API] Invalid video_file type: %s", video_mime)
        raise HTTPException(status_code=400, detail="video_file must be a video type.")
    if not image_mime or not image_mime.startswith("image/"):
        logging.error("[API] Invalid reference_image type: %s", image_mime)
        raise HTTPException(status_code=400, detail="reference_image must be an image type.")

    # Read uploaded files into memory (synchronously)
    video_bytes = video_file.file.read()
    image_bytes = reference_image.file.read()
    logging.info(f"[API] Read files into memory: video={len(video_bytes)} bytes, image={len(image_bytes)} bytes")

    # Call the service for video verification (pass bytes instead of file paths)
    logging.info("[API] Calling video_verification service...")
    result = video_verification(video_bytes, image_bytes, transcribe_reference)
    logging.info("[API] video_verification service returned.")

    if "error" in result:
        logging.error(f"[API] Service returned error: {result['error']}")
        raise HTTPException(status_code=400, detail=result["error"])
    logging.info("[API] Returning response to client.")
    return JSONResponse(content=result) 

@app.get("/health", tags=["Health"], summary="Health Check", description="Returns status ok if the service is healthy.")
def health_check():
    """Health check endpoint for container and load balancer health monitoring."""
    return {"status": "ok"} 