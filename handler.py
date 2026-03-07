"""
RunPod Serverless Handler for IDM-VTON Virtual Try-On
=====================================================
Models loaded ONCE at container startup.

Input:  {
    "human_image_base64": "...",     # Person photo (base64)
    "garment_image_base64": "...",   # Garment photo (base64)
    "category": "upper_body",        # upper_body | lower_body | dresses
    "num_inference_steps": 30,       # 20-50, default 30
    "guidance_scale": 2.0,           # 1.5-3.0, default 2.0
    "seed": 42                       # -1 for random
}
Output: {
    "result_image_base64": "...",    # Result image (base64 PNG)
    "inference_ms": 5200
}
"""

import runpod
import base64
import os
import sys
import time
import io
import gc
import json
import numpy as np
from PIL import Image

# ── Setup paths ──
WORKSPACE = "/workspace"
VTON_DIR = f"{WORKSPACE}/IDM-VTON"
CKPT_DIR = f"{VTON_DIR}/ckpt"

sys.path.insert(0, VTON_DIR)
os.chdir(VTON_DIR)

# ── Patch torch.load for compatibility ──
import torch
_orig_load = torch.load
def _patched_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)
torch.load = _patched_load

# ══════════════════════════════════════════
# STARTUP: Download models & load pipeline
# ══════════════════════════════════════════
print("[Init] Starting IDM-VTON model loading...")
t0 = time.time()

# Download checkpoint files if not cached
def download_ckpts():
    """Download preprocessing model checkpoints"""
    import subprocess
    
    os.makedirs(f"{CKPT_DIR}/densepose", exist_ok=True)
    os.makedirs(f"{CKPT_DIR}/humanparsing", exist_ok=True)
    os.makedirs(f"{CKPT_DIR}/openpose/ckpts", exist_ok=True)
    
    files = {
        f"{CKPT_DIR}/densepose/model_final_162be9.pkl":
            "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/densepose/model_final_162be9.pkl",
        f"{CKPT_DIR}/humanparsing/parsing_atr.onnx":
            "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_atr.onnx",
        f"{CKPT_DIR}/humanparsing/parsing_lip.onnx":
            "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_lip.onnx",
        f"{CKPT_DIR}/openpose/ckpts/body_pose_model.pth":
            "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/openpose/ckpts/body_pose_model.pth",
    }
    
    for path, url in files.items():
        if not os.path.exists(path):
            print(f"[Init] Downloading {os.path.basename(path)}...")
            subprocess.run(["wget", "-q", "-O", path, url], check=True)
        else:
            print(f"[Init] ✅ {os.path.basename(path)} cached")

download_ckpts()

# Load the main pipeline
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import (
    AutoTokenizer, 
    CLIPImageProcessor, 
    CLIPVisionModelWithProjection, 
    CLIPTextModel, 
    CLIPTextModelWithProjection
)

print("[Init] Loading IDM-VTON pipeline from HuggingFace...")

# Import custom UNets from IDM-VTON
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel as UNet2DConditionModel_tryon
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

# Use local path if available (pre-downloaded in Docker), fallback to HF
LOCAL_MODEL = f"{VTON_DIR}/models/idm-vton"
MODEL_ID = LOCAL_MODEL if os.path.exists(LOCAL_MODEL) else "yisol/IDM-VTON"
print(f"[Init] Model source: {MODEL_ID}")

# Load tokenizers
tokenizer_one = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer", use_fast=False)
tokenizer_two = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer_2", use_fast=False)

# Load text encoders
text_encoder_one = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=DTYPE).to(DEVICE)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(MODEL_ID, subfolder="text_encoder_2", torch_dtype=DTYPE).to(DEVICE)

# Load image encoder
image_encoder = CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, subfolder="image_encoder", torch_dtype=DTYPE).to(DEVICE)

# Load VAE
vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=DTYPE).to(DEVICE)

# Load UNets
unet = UNet2DConditionModel_tryon.from_pretrained(MODEL_ID, subfolder="unet", torch_dtype=DTYPE).to(DEVICE)
unet_encoder = UNet2DConditionModel_ref.from_pretrained(MODEL_ID, subfolder="unet_encoder", torch_dtype=DTYPE).to(DEVICE)

# Load scheduler
noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

# Build pipeline
pipe = TryonPipeline.from_pretrained(
    MODEL_ID,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    torch_dtype=DTYPE,
)
pipe.unet_encoder = unet_encoder
pipe = pipe.to(DEVICE)

# Load preprocessing modules
# OpenPose
from preprocess.openpose.run_openpose import OpenPose
openpose_model = OpenPose(0)  # GPU 0
openpose_model.preprocessor.body_estimation.model.to(DEVICE)

# Human parsing
from preprocess.humanparsing.run_parsing import Parsing
parsing_model = Parsing(0)  # GPU 0

# DensePose
from preprocess.detectron2.projects.DensePose.apply_net_gradio import DensePose
densepose_model = DensePose(DEVICE)

# Auto masker 
from utils_mask import get_mask_location

print(f"[Init] ✅ ALL MODELS READY in {time.time()-t0:.1f}s")


# ══════════════════════════════════════════
# PREPROCESSING FUNCTIONS
# ══════════════════════════════════════════

def preprocess_human(human_img_pil, category="upper_body"):
    """Generate all preprocessing artifacts from human image"""
    
    # Resize to standard size
    human_img = human_img_pil.resize((768, 1024))
    
    # Generate OpenPose keypoints
    keypoints = openpose_model(human_img.resize((384, 512)))
    
    # Generate human parsing
    model_parse, _ = parsing_model(human_img.resize((384, 512)))
    
    # Generate mask from parsing
    mask, mask_gray = get_mask_location("hd", category, model_parse, keypoints)
    mask = mask.resize((768, 1024))
    mask_gray = mask_gray.resize((768, 1024))
    
    # Masked human image (agnostic)
    human_img_arg = np.array(human_img)
    mask_arr = np.array(mask_gray)
    # Where mask is white (255), blank out the human
    human_img_arg[mask_arr > 128] = [127, 127, 127]  # gray fill
    human_img_agnostic = Image.fromarray(human_img_arg)
    
    # Generate DensePose
    dense_img = densepose_model(human_img.resize((384, 512)))
    dense_img = dense_img.resize((768, 1024))
    
    return human_img, mask, human_img_agnostic, dense_img


def run_tryon(human_img_pil, garment_img_pil, category="upper_body", 
              num_steps=30, guidance_scale=2.0, seed=42):
    """Run the virtual try-on pipeline"""
    
    # Preprocess human image
    human_img, mask, human_img_agnostic, dense_img = preprocess_human(
        human_img_pil, category
    )
    
    # Resize garment
    garment_img = garment_img_pil.resize((768, 1024))
    
    # Encode prompt (empty prompt for try-on)
    prompt = "model is wearing a garment"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
    
    # Tokenize
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
        pipe.encode_prompt(
            prompt,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )
    
    # CLIP image processing for garment
    clip_processor = CLIPImageProcessor()
    clip_image = clip_processor(images=garment_img, return_tensors="pt").pixel_values
    
    # Generate
    generator = torch.Generator(DEVICE).manual_seed(seed) if seed >= 0 else None
    
    images = pipe(
        prompt_embeds=prompt_embeds.to(DEVICE, DTYPE),
        negative_prompt_embeds=negative_prompt_embeds.to(DEVICE, DTYPE),
        pooled_prompt_embeds=pooled_prompt_embeds.to(DEVICE, DTYPE),
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(DEVICE, DTYPE),
        num_inference_steps=num_steps,
        generator=generator,
        strength=1.0,
        pose_img=dense_img.convert("RGB"),
        text_embeds_cloth=clip_image.to(DEVICE, DTYPE),
        cloth=garment_img.convert("RGB"),
        mask_image=mask,
        image=human_img_agnostic.convert("RGB"),
        height=1024,
        width=768,
        ip_adapter_image=garment_img.convert("RGB"),
        guidance_scale=guidance_scale,
    )[0]
    
    return images[0]


# ══════════════════════════════════════════
# HANDLER
# ══════════════════════════════════════════

def handler(job):
    """RunPod serverless handler"""
    t_start = time.time()
    inp = job.get("input", {})
    
    # Validate input
    human_b64 = inp.get("human_image_base64")
    garment_b64 = inp.get("garment_image_base64")
    
    if not human_b64:
        return {"error": "human_image_base64 is required"}
    if not garment_b64:
        return {"error": "garment_image_base64 is required"}
    
    category = inp.get("category", "upper_body")
    if category not in ["upper_body", "lower_body", "dresses"]:
        category = "upper_body"
    
    num_steps = min(max(int(inp.get("num_inference_steps", 30)), 10), 50)
    guidance_scale = float(inp.get("guidance_scale", 2.0))
    seed = int(inp.get("seed", 42))
    
    try:
        # Decode images
        human_img = Image.open(io.BytesIO(base64.b64decode(human_b64))).convert("RGB")
        garment_img = Image.open(io.BytesIO(base64.b64decode(garment_b64))).convert("RGB")
        
        print(f"[Handler] Human: {human_img.size}, Garment: {garment_img.size}, "
              f"Category: {category}, Steps: {num_steps}, Seed: {seed}")
        
        # Run inference
        result_img = run_tryon(
            human_img, garment_img,
            category=category,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        
        # Encode result as base64 PNG
        buf = io.BytesIO()
        result_img.save(buf, format="PNG", optimize=True)
        result_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        inference_ms = int((time.time() - t_start) * 1000)
        print(f"[Handler] ✅ Done in {inference_ms}ms")
        
        # Cleanup
        del human_img, garment_img, result_img
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "result_image_base64": result_b64,
            "inference_ms": inference_ms
        }
        
    except Exception as e:
        import traceback
        print(f"[Handler] ❌ Error: {e}")
        traceback.print_exc()
        return {"error": str(e)}


# ══════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════
print("[Start] Launching IDM-VTON Serverless handler...")
runpod.serverless.start({"handler": handler})
