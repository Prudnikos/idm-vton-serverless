FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ffmpeg libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Clone IDM-VTON
RUN git clone https://github.com/yisol/IDM-VTON.git /workspace/IDM-VTON

WORKDIR /workspace/IDM-VTON

# Install Python dependencies
RUN pip install --no-cache-dir \
    diffusers==0.25.0 \
    transformers==4.36.2 \
    accelerate==0.25.0 \
    safetensors \
    opencv-python-headless \
    Pillow \
    onnxruntime-gpu \
    scipy \
    scikit-image \
    runpod \
    huggingface_hub

# Install detectron2 for DensePose
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

# Download preprocessing checkpoints
RUN mkdir -p ckpt/densepose ckpt/humanparsing ckpt/openpose/ckpts \
    && wget -q -O ckpt/densepose/model_final_162be9.pkl \
        "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/densepose/model_final_162be9.pkl" \
    && wget -q -O ckpt/humanparsing/parsing_atr.onnx \
        "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_atr.onnx" \
    && wget -q -O ckpt/humanparsing/parsing_lip.onnx \
        "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_lip.onnx" \
    && wget -q -O ckpt/openpose/ckpts/body_pose_model.pth \
        "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/openpose/ckpts/body_pose_model.pth"

# Pre-download HuggingFace model weights (baked into image for fast cold start)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('yisol/IDM-VTON', local_dir='/workspace/IDM-VTON/models/idm-vton'); print('IDM-VTON weights downloaded')"

# Copy handler
COPY handler.py /workspace/IDM-VTON/handler.py

# Set environment
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/hf_cache

CMD ["python", "/workspace/IDM-VTON/handler.py"]
