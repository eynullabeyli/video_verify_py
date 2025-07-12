from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from app.services.liveness_service import liveness_and_similarity
import shutil
import os
import uuid
import mimetypes

app = FastAPI()

@app.post("/liveness-and-similarity-check")
async def liveness_and_similarity_check(
    video_file: UploadFile = File(...),
    reference_image: UploadFile = File(...)
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

    # Call the service for liveness and similarity check
    result = liveness_and_similarity(video_path, ref_path)

    # Clean up uploaded files
    os.remove(video_path)
    os.remove(ref_path)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return JSONResponse(content=result) 