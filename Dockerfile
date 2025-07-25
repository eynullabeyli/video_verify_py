# Use official Python image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the port
EXPOSE 8000

# Explicitly disable GPU usage for all libraries
ENV CUDA_VISIBLE_DEVICES=""

# Production: single process per container (recommended)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# For development, you can use the following (uncomment to use):
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 
#uvicorn main:app --host 0.0.0.0 --port 8000 --reload