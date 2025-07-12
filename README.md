# Video Verification API

This API provides video liveness check, face similarity, and Azerbaijani video transcription using DeepFace, Whisper, and FastAPI.

## Features
- Liveness check from video
- Face similarity check with reference image
- Video-to-text transcription (Azerbaijani, Whisper large model)
- Transcription similarity check with reference text

## Requirements
- Python 3.8+
- ffmpeg (system binary)

## Installation
1. Clone the repository:
   ```sh
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```
2. Install Python dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Install ffmpeg:
   - **macOS:** `brew install ffmpeg`
   - **Ubuntu:** `sudo apt-get install ffmpeg`
   - **Windows:** [Download from ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

## Usage
1. Start the API server:
   ```sh
   uvicorn main:app --reload
   ```
2. Open your browser at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for Swagger UI.
3. Use the `/video-verification-check` endpoint to upload a video, reference image, and (optionally) a reference text for transcription similarity.

## Notes
- The API uses the Whisper `large` model for best transcription accuracy (downloads on first use).
- Default transcription reference: `Mən Nicat Soltanov kredit almaq istəyirəm`
- All temp files are cleaned up automatically.

## Example Request
- Upload a video file (Azerbaijani speech) and a reference image.
- Optionally, provide a reference text for transcription similarity.
- The response includes liveness, similarity, transcription, and transcription similarity percentage.

---

**For any issues or improvements, please open an issue or pull request.** 