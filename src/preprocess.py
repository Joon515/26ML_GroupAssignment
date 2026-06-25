from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import cv2
from PIL import Image


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int = 224
    normalize: bool = True
    random_rotation: bool = True
    clahe: bool = True
    rotation_degrees: float = 20.0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_square(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.Resampling.BILINEAR)


def random_rotate(image: Image.Image, degrees: float, rng: random.Random | None = None) -> Image.Image:
    generator = rng if rng is not None else random
    angle = generator.uniform(-degrees, degrees)
    return image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))

def apply_clahe(image: Image.Image, clip_limit: float, tile_grid_size: int) -> Image.Image:
    rgb = np.asarray(image)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    lab = cv2.merge((clahe.apply(l_channel), a_channel, b_channel))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)


def normalize_array(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (arr - mean) / std


def denormalize_array(arr: np.ndarray) -> Image.Image:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    pixels = np.clip((arr * std + mean) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


def preprocess_image(image: Image.Image, config: PreprocessConfig, *, train: bool,
                     rng: random.Random | None = None) -> np.ndarray:
    out = image
    out = resize_square(out, config.image_size)
    if train and config.random_rotation:
        out = random_rotate(out, config.rotation_degrees, rng)
    if config.clahe:
        out = apply_clahe(out, config.clahe_clip_limit, config.clahe_tile_grid_size)
    if config.normalize:
        return normalize_array(out)
    return np.asarray(out, dtype=np.float32) / 255.0


def preprocess_stages(image: Image.Image, config: PreprocessConfig, rng: random.Random | None = None) -> list[tuple[str, Image.Image]]:
    stages: list[tuple[str, Image.Image]] = [("original", image.copy())]
    out = image
    out = resize_square(out, config.image_size)
    stages.append(("resize", out.copy()))
    if config.random_rotation:
        out = random_rotate(out, config.rotation_degrees, rng)
        stages.append(("random_rotation", out.copy()))
    if config.clahe:
        out = apply_clahe(out, config.clahe_clip_limit, config.clahe_tile_grid_size)
        stages.append(("clahe", out.copy()))
    if config.normalize:
        stages.append(("normalize", denormalize_array(normalize_array(out))))
    return stages


def config_from_enabled(enabled: Iterable[str], image_size: int = 224) -> PreprocessConfig:
    active = set(enabled)
    return PreprocessConfig(
        image_size=image_size,
        normalize="normalize" in active,
        random_rotation="rotate" in active,
        clahe="clahe" in active,
    )


ALL_PREPROCESS_STEPS: tuple[str, ...] = ("normalize", "rotate", "clahe")
