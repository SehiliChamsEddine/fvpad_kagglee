"""
The five pre-trained CNNs used as *frozen feature extractors*.

Each network is cut before its ImageNet classifier. An image goes through the
convolutional part, the last feature map is global-average-pooled (GAP) and
the result is the image's embedding (a fixed-length vector):

    image -> preprocessing -> conv layers -> last feature map (C x h x w)
          -> global average pooling -> embedding in R^C

| project name | backbone        | weights (ImageNet-1k)                 | input   | dim  |
|--------------|-----------------|---------------------------------------|---------|------|
| TransferNN   | EfficientNet-B0 | timm `efficientnet_b0.ra_in1k`        | 224x224 | 1280 |
| VGGKNN       | VGG16           | original Oxford/Keras VGG16           | 224x224 | 512  |
| TinyKNN      | TinyNet-A       | Huawei release used by timm           | 192x192 | 1280 |
| MobileKNN    | MobileNetV2 1.0 | timm `mobilenetv2_100.ra_in1k`        | 224x224 | 1280 |
| SqueezeKNN   | SqueezeNet 1.1  | DeepScale weights (Keras port)        | 224x224 | 512  |

All weight files are downloaded from fixed GitHub release URLs and checked
against a SHA-256 hash, so everyone who runs the code uses the identical
weights. VGG16 and SqueezeNet 1.1 are stored in Keras HDF5 format; they are
converted to the equivalent PyTorch layers below (kernel layout HWIO -> OIHW).
`scripts/verify_weights.py` checks these conversions by classifying a test
photo with the full (classifier-included) versions of the same weights.
"""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision
from PIL import Image

from . import config

# ImageNet statistics used by the timm models (RGB, values in [0, 1]).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# "Caffe" preprocessing used by the original VGG16 / SqueezeNet weights:
# BGR channel order, values in [0, 255], per-channel mean subtracted, no scaling.
CAFFE_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)


@dataclass(frozen=True)
class ModelSpec:
    name: str              # name used in this project
    backbone: str          # architecture name
    source: str            # "timm" (PyTorch checkpoint) or "keras" (HDF5)
    url: str               # where the weights are downloaded from
    sha256: str            # expected hash of the weight file
    input_size: int        # square input resolution the weights were trained at
    embed_dim: int         # length of the embedding (channels of last map)
    preprocessing: str     # "imagenet" or "caffe"

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


_GH_TIMM = "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/"

MODEL_SPECS: dict[str, ModelSpec] = {
    "TransferNN": ModelSpec(
        "TransferNN", "efficientnet_b0", "timm",
        _GH_TIMM + "efficientnet_b0_ra-3dd342df.pth",
        "3dd342dfa1fee25ae65e7bbdf8998cad6e45d6e77e69d580f0bd14d3eeb0b3f3",
        224, 1280, "imagenet"),
    "VGGKNN": ModelSpec(
        "VGGKNN", "vgg16", "keras",
        "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/"
        "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "bfe5187d0a272bed55ba430631598124cff8e880b98d38c9e56c8d66032abdc1",
        224, 512, "caffe"),
    "TinyKNN": ModelSpec(
        "TinyKNN", "tinynet_a", "timm",
        "https://github.com/huawei-noah/CV-Backbones/releases/download/v1.2.0/tinynet_a.pth",
        "170b3d8ea1c2495b3fb2ca74b71a5acb26c2345d13cb97c5ffd77f6a9b7ca568",
        192, 1280, "imagenet"),
    "MobileKNN": ModelSpec(
        "MobileKNN", "mobilenetv2_100", "timm",
        _GH_TIMM + "mobilenetv2_100_ra-b33bc2c4.pth",
        "b33bc2c4319407fabd894326c3c758959c2c21fbc1cbac888cad6aa9f017cf2a",
        224, 1280, "imagenet"),
    "SqueezeKNN": ModelSpec(
        "SqueezeKNN", "squeezenet1_1", "keras",
        "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/"
        "squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "2d935d8c102ad1232484a77da8e6fbdf3ff4ab608974a0d8b6283115ffff204f",
        224, 512, "caffe"),
}


# --------------------------------------------------------------------------
# Weight download
# --------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_weights(spec: ModelSpec, weights_dir: Path = config.WEIGHTS_DIR) -> Path:
    """Download the weight file once and verify its SHA-256 hash."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    path = weights_dir / spec.filename
    if not path.exists():
        print(f"[weights] downloading {spec.filename} ...")
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(spec.url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(path)
    digest = _sha256(path)
    if digest != spec.sha256:
        raise RuntimeError(f"Hash mismatch for {path}: {digest} != {spec.sha256}")
    return path


# --------------------------------------------------------------------------
# Keras HDF5 helpers
# --------------------------------------------------------------------------
def _keras_layer_params(h5: h5py.File, layer: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (kernel, bias) stored under a Keras layer group."""
    arrays: list[np.ndarray] = []
    h5[layer].visititems(lambda n, o: arrays.append(o[()]) if isinstance(o, h5py.Dataset) else None)
    kernel = next(a for a in arrays if a.ndim == 4)
    bias = next(a for a in arrays if a.ndim == 1)
    return kernel, bias


def _load_keras_conv(conv: nn.Conv2d, kernel: np.ndarray, bias: np.ndarray) -> None:
    """Copy a Keras conv kernel (H, W, in, out) into a PyTorch conv (out, in, H, W)."""
    w = torch.from_numpy(np.ascontiguousarray(kernel.transpose(3, 2, 0, 1)))
    if tuple(w.shape) != tuple(conv.weight.shape):
        raise ValueError(f"shape mismatch {tuple(w.shape)} vs {tuple(conv.weight.shape)}")
    conv.weight.data.copy_(w)
    conv.bias.data.copy_(torch.from_numpy(bias))


# --------------------------------------------------------------------------
# Backbones
# --------------------------------------------------------------------------
class GAPBackbone(nn.Module):
    """Convolutional feature extractor followed by global average pooling."""

    def __init__(self, features: nn.Module):
        super().__init__()
        self.features = features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).mean(dim=(2, 3))


# Indices of the 13 conv layers inside torchvision's vgg16().features and the
# matching Keras layer names (same order).
VGG16_CONV_IDX = (0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28)
VGG16_KERAS_NAMES = (
    "block1_conv1", "block1_conv2", "block2_conv1", "block2_conv2",
    "block3_conv1", "block3_conv2", "block3_conv3",
    "block4_conv1", "block4_conv2", "block4_conv3",
    "block5_conv1", "block5_conv2", "block5_conv3")

# Fire modules inside torchvision's squeezenet1_1().features.
SQUEEZE_FIRE_IDX = {"fire2": 3, "fire3": 4, "fire4": 6, "fire5": 7,
                    "fire6": 9, "fire7": 10, "fire8": 11, "fire9": 12}


def build_vgg16_from_keras(h5_path: Path) -> nn.Module:
    features = torchvision.models.vgg16(weights=None).features
    with h5py.File(h5_path, "r") as h5:
        for idx, name in zip(VGG16_CONV_IDX, VGG16_KERAS_NAMES):
            _load_keras_conv(features[idx], *_keras_layer_params(h5, name))
    return GAPBackbone(features)


def build_squeezenet11_from_keras(h5_path: Path) -> nn.Module:
    features = torchvision.models.squeezenet1_1(weights=None).features
    with h5py.File(h5_path, "r") as h5:
        _load_keras_conv(features[0], *_keras_layer_params(h5, "conv1"))
        for fire, idx in SQUEEZE_FIRE_IDX.items():
            m = features[idx]
            _load_keras_conv(m.squeeze, *_keras_layer_params(h5, f"{fire}/squeeze1x1"))
            _load_keras_conv(m.expand1x1, *_keras_layer_params(h5, f"{fire}/expand1x1"))
            _load_keras_conv(m.expand3x3, *_keras_layer_params(h5, f"{fire}/expand3x3"))
    return GAPBackbone(features)


def build_timm_backbone(arch: str, ckpt_path: Path) -> nn.Module:
    # num_classes=0 removes the classifier; timm then returns the pooled
    # (GAP) output of the last conv stage, i.e. exactly the embedding we want.
    model = timm.create_model(arch, pretrained=False, num_classes=0)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{arch}: missing={missing} unexpected={unexpected}")
    return model


def load_model(name: str) -> nn.Module:
    """Build one of the five frozen feature extractors (eval mode, no grads)."""
    spec = MODEL_SPECS[name]
    path = ensure_weights(spec)
    if spec.backbone == "vgg16":
        model = build_vgg16_from_keras(path)
    elif spec.backbone == "squeezenet1_1":
        model = build_squeezenet11_from_keras(path)
    else:
        model = build_timm_backbone(spec.backbone, path)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def load_gray(path: str | Path) -> Image.Image:
    """Read an image as 8-bit grayscale (finger-vein images are single-channel)."""
    with Image.open(path) as im:
        return im.convert("L")


def preprocess(img: Image.Image, spec: ModelSpec) -> np.ndarray:
    """Turn a grayscale PIL image into the CHW float array a model expects.

    1. Resize the *whole* image (no cropping, so no part of the finger is lost)
       to the square resolution the network was trained at, with bicubic,
       anti-aliased interpolation.
    2. Copy the gray channel into 3 channels (ImageNet models expect colour).
    3. Apply the normalisation the weights were trained with.
    """
    s = spec.input_size
    g = np.asarray(img.resize((s, s), Image.BICUBIC), dtype=np.float32)
    x = np.repeat(g[None, :, :], 3, axis=0)                 # 3 x s x s, [0, 255]
    if spec.preprocessing == "imagenet":                    # RGB, [0,1], mean/std
        x = (x / 255.0 - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    elif spec.preprocessing == "caffe":                     # BGR, [0,255], -mean
        x = x[::-1] - CAFFE_MEAN_BGR[:, None, None]
    else:
        raise ValueError(spec.preprocessing)
    return np.ascontiguousarray(x, dtype=np.float32)
