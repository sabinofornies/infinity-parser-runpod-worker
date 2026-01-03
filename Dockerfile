# Infinity-Parser-7B RunPod Serverless Worker
# Based on: https://huggingface.co/infly/Infinity-Parser-7B

# Use smaller runtime image instead of 8GB devel image
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (no flash-attn to avoid 1h+ compile time)
RUN pip install --no-cache-dir \
    runpod \
    transformers>=4.45.0 \
    accelerate \
    torch \
    torchvision \
    qwen-vl-utils \
    pdf2image \
    Pillow

# Model downloads at runtime (first cold start will be slower, but build succeeds)
# Build server has limited RAM and can't load 7B model during build

# Copy handler
COPY handler.py /app/handler.py

# Start the handler
CMD ["python3", "-u", "/app/handler.py"]
