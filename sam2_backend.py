import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import io
import cv2
import json

class SAM2Segmenter:
    def __init__(self, sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt", model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml"):
        self.sam2_checkpoint = sam2_checkpoint
        self.model_cfg = model_cfg
        self.frame_masks = {}

        self.device = self._get_device()
        self.predictor = self._build_predictor()

        self.inference_states = {}


    def _get_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")
        if device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        return device
    
    def _build_predictor(self):
        predictor = build_sam2_video_predictor(self.model_cfg, self.sam2_checkpoint, device=self.device)
        return predictor
    
    def load_video(self, folder_path: str):
        if folder_path not in self.inference_states:
            print(f"Initializing SAM2 inference state for folder: {folder_path}")
            state = self.predictor.init_state(video_path=folder_path)
            self.inference_states[folder_path] = state
        return self.inference_states[folder_path]