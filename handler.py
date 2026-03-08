"""
RunPod Serverless Handler for IDM-VTON Virtual Try-On
=====================================================
Based on IDM-VTON gradio_demo/app.py

Input:  {
    "human_image_base64": "...",
    "garment_image_base64": "...",
    "garment_description": "a shirt",
    "category": "upper_body",
    "num_inference_steps": 30,
    "guidance_scale": 2.0,
    "seed": 42
}
Output: {
    "result_image_base64": "...",
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
import numpy as np
from PIL import Image

# ── Setup paths ──
VTON_DIR = "/workspace/IDM-VTON"
sys.path.insert(0, VTON_DIR)
sys.path.insert(0, os.path.join(VTON_DIR, "gradio_demo"))
os.chdir(VTON_DIR)

# ── Patch torch.load ──
import torch
_orig_load = torch.load
def _patched_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)
torch.load = _patched_load

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float16

# ══════════════════════════════════════════
# STARTUP: Load models ONCE
# ══════════════════════════════════════════
print("[Init] Starting IDM-VTON model loading...")
t0 = time.time()

from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    AutoTokenizer,
)
from diffusers import DDPMScheduler, AutoencoderKL
from utils_mask import get_mask_location
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation

# Use local path if available (pre-downloaded in Docker), fallback to HF
LOCAL_MODEL = os.path.join(VTON_DIR, "models", "idm-vton")
MODEL_ID = LOCAL_MODEL if os.path.exists(LOCAL_MODEL) else "yisol/IDM-VTON"
print(f"[Init] Model source: {MODEL_ID}")

# Load base pipeline
base_path = 'yisol/IDM-VTON'
unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", torch_dtype=DTYPE)
unet.requires_grad_(False)

tokenizer_one = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer", use_fast=False)
tokenizer_two = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer_2", use_fast=False)
noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
text_encoder_one = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=DTYPE)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(MODEL_ID, subfolder="text_encoder_2", torch_dtype=DTYPE)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, subfolder="image_encoder", torch_dtype=DTYPE)
vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=DTYPE)

# UNet Encoder
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(MODEL_ID, subfolder="unet_encoder", torch_dtype=DTYPE)
UNet_Encoder.requires_grad_(False)

# Build pipeline
pipe = TryonPipeline.from_pretrained(
    base_path,
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
pipe.unet_encoder = UNet_Encoder
pipe = pipe.to(DEVICE)

# Load preprocessing models
parsing_model = Parsing(0)
openpose_model = OpenPose(0)
openpose_model.preprocessor.body_estimation.model.to(DEVICE)

print(f"[Init] ✅ ALL MODELS READY in {time.time()-t0:.1f}s")

# Transforms
tensor_transfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


# ══════════════════════════════════════════
# INFERENCE FUNCTION (based on app.py start_tryon)
# ══════════════════════════════════════════

def run_tryon(human_img_pil, garment_img_pil, garment_des="a garment",
              category="upper_body", num_steps=30, guidance_scale=2.0, seed=42):
    """Run virtual try-on based on app.py logic"""

    # Resize inputs
    human_img = human_img_pil.resize((768, 1024))
    garm_img = garment_img_pil.resize((768, 1024))

    # OpenPose
    keypoints = openpose_model(human_img.resize((384, 512)))

    # Human parsing
    model_parse, _ = parsing_model(human_img.resize((384, 512)))

    # Get mask
    mask, mask_gray = get_mask_location('hd', category, model_parse, keypoints)
    mask = mask.resize((768, 1024))
    mask_gray = (mask_gray + 1.0) / 2.0
    mask_gray = to_pil_image(mask_gray)
    mask_gray = mask_gray.resize((768, 1024))

    # DensePose
    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
    args = apply_net.create_argument_parser().parse_args((
        'show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml',
        './ckpt/densepose/model_final_162be9.pkl', 'dp_segm',
        '-v', '--opts', 'MODEL.DEVICE', 'cuda'
    ))
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize((768, 1024))

    # Prepare images
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            prompt = "model is wearing " + garment_des
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

            with torch.inference_mode():
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = pipe.encode_prompt(
                    prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )

                prompt_c = "a photo of " + garment_des
                (
                    prompt_embeds_c,
                    _,
                    _,
                    _,
                ) = pipe.encode_prompt(
                    prompt_c,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=negative_prompt,
                )

            # Prepare garment tensor
            garm_tensor = tensor_transfm(garm_img).unsqueeze(0).to(DEVICE, DTYPE)
            generator = torch.Generator(DEVICE).manual_seed(seed) if seed >= 0 else None

            # Run pipeline
            images = pipe(
                prompt_embeds=prompt_embeds.to(DEVICE, DTYPE),
                negative_prompt_embeds=negative_prompt_embeds.to(DEVICE, DTYPE),
                pooled_prompt_embeds=pooled_prompt_embeds.to(DEVICE, DTYPE),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(DEVICE, DTYPE),
                num_inference_steps=num_steps,
                generator=generator,
                strength=1.0,
                pose_img=tensor_transfm(pose_img).unsqueeze(0).to(DEVICE, DTYPE),
                text_embeds_cloth=prompt_embeds_c.to(DEVICE, DTYPE),
                cloth=garm_tensor,
                mask_image=mask,
                image=human_img,
                height=1024,
                width=768,
                ip_adapter_image=garm_img.resize((768, 1024)),
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

    garment_des = inp.get("garment_description", "a garment")
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
            garment_des=garment_des,
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
