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
   git clone https://github.com/eynullabeyli/video_verify_py.git
   cd video_verify_py
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
- The API uses the Whisper `medium` model for best transcription accuracy (downloads on first use).
- Default transcription reference: `Mən Nicat Soltanov kredit almaq istəyirəm`
- All temp files are cleaned up automatically.

## Docker Deployment

This project is production-ready with Docker Compose and Nginx for load balancing and scaling.

### Build and Start (with Scaling)
```sh
docker compose down  # Stop and remove any running containers

docker compose up -d --scale video-verification-api=3  # Build and start with 3 FastAPI containers
```
- Nginx will be available at [http://localhost:8000](http://localhost:8000)
- You can adjust the number of FastAPI containers by changing the `--scale` value.

**Note:** The Docker container is explicitly configured to use only CPU and RAM. GPU usage is disabled by setting the environment variable `CUDA_VISIBLE_DEVICES=""` inside the container. No GPU or CUDA drivers are required or used.

### Health Checks
- The API exposes a `/health` endpoint for container and load balancer health monitoring.
- Docker Compose and Nginx use this endpoint to check service health.

### Resource Usage
- The FastAPI containers are configured to use all available CPU cores on your machine for maximum performance.
- Temporary files are stored in a Docker volume mounted at `/tmp`.

### Large File Uploads
- Nginx is configured to allow uploads up to 100MB (`client_max_body_size 100M`).
- Nginx proxy and read timeouts are set to 10 minutes to support long-running requests.

### Troubleshooting
- If you see `502 Bad Gateway`, the app containers may not be ready yet. Wait a few seconds and try again.
- If you see `413 Request Entity Too Large`, increase `client_max_body_size` in `nginx.conf`.
- If you see `504 Gateway Time-out`, increase the timeout values in `nginx.conf` or optimize your backend.

## Example Request
- Upload a video file (Azerbaijani speech) and a reference image.
- Optionally, provide a reference text for transcription similarity.
- The response includes liveness, similarity, transcription, and transcription similarity percentage.

---

**For any issues or improvements, please open an issue or pull request.**