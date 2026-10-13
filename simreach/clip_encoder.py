"""Frozen CLIP vision and text encoder for the VLA action head.

Weights download on first construction (``openai/clip-vit-base-patch32``). Tests
inject a stub model and processor so CI does not fetch the checkpoint.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import CLIPModel, CLIPProcessor

DEFAULT_CLIP_NAME = "openai/clip-vit-base-patch32"


class _Processor(Protocol):
    def __call__(
        self,
        text: list[str],
        images: list[Image.Image],
        return_tensors: str = "pt",
        padding: bool = True,
    ) -> dict[str, torch.Tensor]: ...


class ClipEncoder(nn.Module):
    """Batch embedder. The backbone is frozen; only the action head is trained."""

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_NAME,
        model: nn.Module | None = None,
        processor: _Processor | None = None,
    ) -> None:
        super().__init__()
        if model is None or processor is None:
            model = CLIPModel.from_pretrained(model_name)
            processor = CLIPProcessor.from_pretrained(model_name)
        self.model = model
        self.processor = processor
        config = getattr(model, "config", None)
        projection = getattr(config, "projection_dim", 512)
        self.out_dim = int(projection)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def embed(self, images_nhwc: np.ndarray, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if len(images_nhwc) != len(texts):
            raise ValueError("images and texts must have the same length")
        pil_images = [Image.fromarray(np.ascontiguousarray(frame)) for frame in images_nhwc]
        inputs = self.processor(text=texts, images=pil_images, return_tensors="pt", padding=True)
        outputs = self.model(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        return outputs.image_embeds, outputs.text_embeds

    def embed_numpy(
        self,
        images_nhwc: np.ndarray,
        texts: list[str],
        batch_size: int = 16,
    ) -> tuple[np.ndarray, np.ndarray]:
        visual_parts: list[np.ndarray] = []
        text_parts: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            stop = start + batch_size
            visual, text = self.embed(images_nhwc[start:stop], texts[start:stop])
            visual_parts.append(visual.detach().cpu().numpy().astype(np.float32))
            text_parts.append(text.detach().cpu().numpy().astype(np.float32))
        if not visual_parts:
            raise ValueError("no rows to embed")
        return np.concatenate(visual_parts, axis=0), np.concatenate(text_parts, axis=0)
