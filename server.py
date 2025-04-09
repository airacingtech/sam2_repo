import base64
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import io
import cv2
import json
import hydra
import shutil
from pydantic import BaseModel
from pathlib import Path
from sam2_backend import SAM2Segmenter
from collections import defaultdict

MASK_STORE = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))

MASK_COLORS = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Cyan
    (255, 165, 0),   # Orange
    (128, 0, 128),   # Purple
    (0, 128, 128),   # Teal
    (128, 128, 0),   # Olive
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# SAM2 Class call
segmenter = SAM2Segmenter()


BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"

# UPLOAD_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# # do not use with cuda
# device = torch.device("cuda")
# #do not remove
# if device.type == "cuda":
#     # use bfloat16 for the entire notebook
#     torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
#     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#     if torch.cuda.get_device_properties(0).major >= 8:
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True
# sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"


# predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

FRONTEND_IMAGES_DIR = "../../frontend/public/image-sequences"  # adjust this path if needed
BACKEND_UPLOADS_DIR = "uploads"

class FolderRequest(BaseModel):
    folder: str

class SegmentRequest(BaseModel):
    frame_index: int
    folder: str
    x: int
    y: int
    is_positive: bool
    object_id: int

class PropagateRequest(BaseModel):
    folder: str
    total_frames: int
    

app.mount("/static", StaticFiles(directory=str(UPLOADS_DIR)), name="static")

@app.get("/api/images")
async def get_images(folder: str):
    folder_path = UPLOADS_DIR / folder

    if not os.path.exists(folder_path):
        return JSONResponse(status_code=404, content={"error": "Folder not found"})

    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    image_urls = [f"/static/{folder}/{file}" for file in image_files]
    # print(f"[API] Scanning folder: {folder_path}")
    # print(f"[API] Found images: {image_files}")


    return {"images": image_urls}

@app.post("/upload-images/")
async def receive_folder_and_copy(req: FolderRequest):
    folder_name = req.folder
    src_folder = os.path.join(FRONTEND_IMAGES_DIR, folder_name)
    dst_folder = UPLOADS_DIR / folder_name  

    if not os.path.exists(src_folder):
        return {"error": f"Source folder '{src_folder}' does not exist."}

    if dst_folder.exists():
        shutil.rmtree(dst_folder)

    shutil.copytree(src_folder, dst_folder)
    # print(f"[DEBUG] Copied to: {dst_folder}")
    # print(f"[DEBUG] Exists? {os.path.exists(dst_folder / '00056.jpg')}")
    # print(f"[DEBUG] Full path: {dst_folder / '00056.jpg'}")
    return {
        "status": "folder copied",
        "source": src_folder,
        "destination": str(dst_folder),
        "file_count": len(os.listdir(dst_folder)),
    }

@app.post("/process-click/")
async def process_click(req: SegmentRequest):
    folder_path = UPLOADS_DIR / req.folder
    if not folder_path.exists():
        raise HTTPException(status_code=404, detail="Folder not found")
    images = sorted([f for f in folder_path.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if not images or req.frame_index >= len(images):
        raise HTTPException(status_code=400, detail="Invalid frame index")
    image_path = images[req.frame_index]
    image = cv2.imread(str(image_path))
    if image is None:
        raise HTTPException(status_code=500, detail="Failed to load image")
    state = segmenter.load_video(str(folder_path))
    point_array = np.array([[req.x, req.y]], dtype=np.float32)
    label_array = np.array([1 if req.is_positive else 0], dtype=np.int32)
    _, out_obj_ids, out_mask_logits = segmenter.predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=req.frame_index,
        obj_id=req.object_id,
        points=point_array,
        labels=label_array,
    )
    for i, out_obj_id in enumerate(out_obj_ids):
        mask_prob = torch.sigmoid(out_mask_logits[i])
        binary_mask = (mask_prob > 0.5).int().squeeze().cpu().numpy().astype(np.uint8)
        MASK_STORE[req.folder][req.frame_index][int(out_obj_id)] = binary_mask

    overlay = image.copy()
    for obj_id, mask in MASK_STORE[req.folder][req.frame_index].items():
        color = MASK_COLORS[obj_id % len(MASK_COLORS)]
        overlay[mask > 0] = color
    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    _, buffer = cv2.imencode(".jpg", result)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

@app.post("/propagate-masks/")
async def propagate_masks(req: PropagateRequest):
    folder = req.folder
    total_frames = req.total_frames

    folder_path = UPLOADS_DIR / folder
    if not folder_path.exists():
        raise HTTPException(status_code=404, detail="Folder not found")

    state = segmenter.load_video(str(folder_path))

    for out_frame_idx, out_obj_ids, out_mask_logits in segmenter.predictor.propagate_in_video(state):
        for i, out_obj_id in enumerate(out_obj_ids):
            mask_prob = torch.sigmoid(out_mask_logits[i])
            binary_mask = (mask_prob > 0.5).int().squeeze().cpu().numpy().astype(np.uint8)
            MASK_STORE[folder][out_frame_idx][int(out_obj_id)] = binary_mask

    segmenter.predictor.reset_state(state)

    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    overlayed_images = []

    for frame_idx, img_name in enumerate(image_files):
        image_path = folder_path / img_name
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        overlay = image.copy()
        for obj_id, mask in MASK_STORE[folder][frame_idx].items():
            if mask is not None:
                color = MASK_COLORS[obj_id % len(MASK_COLORS)]
                overlay[mask > 0] = color
        result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        _, buffer = cv2.imencode(".jpg", result)
        base64_image = base64.b64encode(buffer).decode("utf-8")
        overlayed_images.append(base64_image)

    return {
        "status": "success",
        "frames_updated": list(MASK_STORE[folder].keys()),
        "images": overlayed_images  # List[str] of base64 jpgs
    }

@app.post("/save-segmentations/")
async def save_segmentations(req: FolderRequest):
    folder = req.folder
    if folder not in MASK_STORE:
        raise HTTPException(status_code=404, detail="No masks found for the given folder")

    segmentations_dir = BASE_DIR / "../../frontend/public/segmentations" / folder
    os.makedirs(segmentations_dir, exist_ok=True)

    for frame_idx, obj_masks in MASK_STORE[folder].items():
        frame_dict = {}
        for obj_id, mask in obj_masks.items():
            if mask is not None:
                # Serialize binary mask to a compact string (or list of 0s and 1s)
                mask_serializable = mask.tolist()
                frame_dict[str(obj_id)] = mask_serializable

        with open(segmentations_dir / f"{frame_idx:05d}.json", "w") as f:
            json.dump(frame_dict, f)

    return {
        "status": "saved",
        "folder": folder,
        "frames_saved": len(MASK_STORE[folder])
    }

