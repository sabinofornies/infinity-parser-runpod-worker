# Infinity-Parser-7B RunPod Serverless Worker
# Based on: https://huggingface.co/infly/Infinity-Parser-7B

# Use PyTorch 2.5 for compatibility with latest transformers
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (no flash-attn to avoid 1h+ compile time)
# Use transformers 4.49.0+ for Qwen2_5_VLForConditionalGeneration support
RUN pip install --no-cache-dir \
    runpod \
    transformers>=4.49.0 \
    accelerate \
    qwen-vl-utils \
    pdf2image \
    Pillow

# Model downloads at runtime (first cold start will be slower, but build succeeds)
# Build server has limited RAM and can't load 7B model during build

# Copy handler
COPY handler.py /app/handler.py

# Start the handler
CMD ["python3", "-u", "/app/handler.py"]
