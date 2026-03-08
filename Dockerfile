FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
SHELL ["/bin/bash", "-c"]
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends git wget ffmpeg libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/yisol/IDM-VTON.git /workspace/IDM-VTON
WORKDIR /workspace/IDM-VTON
RUN pip install --no-cache-dir numpy==1.26.4 diffusers==0.25.0 transformers==4.36.2 accelerate==0.25.0 huggingface_hub==0.23.4 einops safetensors opencv-python-headless Pillow onnxruntime-gpu scipy scikit-image runpod av
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'
RUN cp gradio_demo/apply_net.py . && cp gradio_demo/utils_mask.py . && cp -r gradio_demo/densepose . 2>/dev/null; true
RUN mkdir -p ckpt/densepose ckpt/humanparsing ckpt/openpose/ckpts && wget -q -O ckpt/densepose/model_final_162be9.pkl "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/densepose/model_final_162be9.pkl" && wget -q -O ckpt/humanparsing/parsing_atr.onnx "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_atr.onnx" && wget -q -O ckpt/humanparsing/parsing_lip.onnx "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_lip.onnx" && wget -q -O ckpt/openpose/ckpts/body_pose_model.pth "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/openpose/ckpts/body_pose_model.pth"
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('yisol/IDM-VTON', local_dir='/workspace/IDM-VTON/models/idm-vton'); print('Done')"
COPY handler.py /workspace/IDM-VTON/handler.py
ENV PYTHONUNBUFFERED=1
CMD ["python", "/workspace/IDM-VTON/handler.py"]
