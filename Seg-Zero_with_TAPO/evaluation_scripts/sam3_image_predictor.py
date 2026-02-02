import torch
import numpy as np
from PIL import Image
import yaml
import os
import sam3

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SAM3ImagePredictor:
    """
    Drop-in replacement for SAM2ImagePredictor.
    Resolves SAM3 config relative to installed sam3 package.
    """

    def __init__(self, segmentation_model_path: str, device: str = "cuda"):
        self.device = device

        # ---- resolve SAM3 config path robustly ----
        sam3_root = os.path.dirname(sam3.__file__)
        candidates = [
            os.path.join(sam3_root, "configs", "sam3_image.yaml"),
            os.path.join(sam3_root, "configs", "sam3", "sam3_image.yaml"),
            os.path.join(sam3_root, "configs", "sam3_hiera_l.yaml"),
        ]

        cfg_path = None
        for p in candidates:
            if os.path.exists(p):
                cfg_path = p
                break

        if cfg_path is None:
            raise FileNotFoundError(
                f"SAM3 config not found. Tried: {candidates}"
            )

        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        # ---- build model ----
        self.model = build_sam3_image_model(
            cfg=cfg,
            checkpoint=segmentation_model_path,
            device=device,
        )
        self.model.eval()

        self.processor = Sam3Processor(self.model)
        self.state = None

    def set_image(self, image: Image.Image):
        self.state = self.processor.set_image(image)

    def predict(self, point_coords=None, point_labels=None, box=None):
        prompts = {}

        if box is not None:
            prompts["boxes"] = torch.tensor(
                [box], dtype=torch.float32, device=self.device
            )

        if point_coords is not None:
            prompts["points"] = (
                torch.tensor(point_coords, dtype=torch.float32, device=self.device),
                torch.tensor(point_labels, dtype=torch.int64, device=self.device),
            )

        with torch.no_grad():
            out = self.processor.predict(
                state=self.state,
                prompts=prompts,
            )

        return (
            out["masks"].cpu().numpy(),
            out["scores"].cpu().numpy(),
            None,
        )
