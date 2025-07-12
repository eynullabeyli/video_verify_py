from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from app.services.liveness_service import liveness_and_similarity
import shutil
import os
import uuid

app = FastAPI()

@app.post("/liveness-check")
async def liveness_check(
    video_file: UploadFile = File(...),
    reference_image: UploadFile = File(...)
):
    # Save uploaded files to tmp directory
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    video_path = os.path.join(tmp_dir, f"{uuid.uuid4()}_{video_file.filename}")
    ref_path = os.path.join(tmp_dir, f"{uuid.uuid4()}_{reference_image.filename}")

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video_file.file, f)
    with open(ref_path, "wb") as f:
        shutil.copyfileobj(reference_image.file, f)

    # Call the service
    result = liveness_and_similarity(video_path, ref_path)

    # Clean up uploaded files
    os.remove(video_path)
    os.remove(ref_path)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return JSONResponse(content=result) 