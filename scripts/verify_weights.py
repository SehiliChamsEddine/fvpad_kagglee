"""
Sanity check for the pretrained weights (run once; not part of the experiments).

The VGG16 and SqueezeNet 1.1 weights are converted from Keras HDF5 files to
PyTorch. A wrong axis order would silently produce meaningless embeddings, so
we load the *classifier-included* versions of the same Keras files, run a
photo of a Samoyed dog (ImageNet class 258) through every network and check
that all five networks recognise it.

Usage:  python scripts/verify_weights.py path/to/dog.jpg  [vgg16_top.h5] [squeezenet_top.h5]
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fvpad import models  # noqa: E402

SAMOYED = 258


def prep_rgb(img: Image.Image, spec: models.ModelSpec) -> torch.Tensor:
    s = spec.input_size
    x = np.asarray(img.convert("RGB").resize((s, s), Image.BICUBIC), dtype=np.float32).transpose(2, 0, 1)
    if spec.preprocessing == "imagenet":
        x = (x / 255.0 - models.IMAGENET_MEAN[:, None, None]) / models.IMAGENET_STD[:, None, None]
    else:
        x = x[::-1] - models.CAFFE_MEAN_BGR[:, None, None]
    return torch.from_numpy(np.ascontiguousarray(x))[None]


def main(dog: str, vgg_top: str | None, sq_top: str | None) -> None:
    img = Image.open(dog)
    import timm
    for name in ("TransferNN", "TinyKNN", "MobileKNN"):
        spec = models.MODEL_SPECS[name]
        m = timm.create_model(spec.backbone, pretrained=False)
        m.load_state_dict(torch.load(models.ensure_weights(spec), map_location="cpu", weights_only=True))
        with torch.no_grad():
            p = m.eval()(prep_rgb(img, spec)).softmax(1)[0]
        print(f"{name:11s} top-1 = {int(p.argmax())}  p(Samoyed) = {float(p[SAMOYED]):.3f}")

    if vgg_top:
        spec = models.MODEL_SPECS["VGGKNN"]
        backbone = models.build_vgg16_from_keras(Path(vgg_top))
        with h5py.File(vgg_top, "r") as h5:
            fc = []
            for layer in ("fc1", "fc2", "predictions"):
                arr = []
                h5[layer].visititems(lambda n, o: arr.append(o[()]) if isinstance(o, h5py.Dataset) else None)
                w = next(a for a in arr if a.ndim == 2); b = next(a for a in arr if a.ndim == 1)
                fc.append((w, b))
        # Keras flattens the 7x7x512 map in (H, W, C) order; PyTorch in (C, H, W).
        w1 = fc[0][0].reshape(7, 7, 512, 4096).transpose(2, 0, 1, 3).reshape(25088, 4096)
        head = nn.Sequential(nn.Flatten(), nn.Linear(25088, 4096), nn.ReLU(), nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, 1000))
        for lin, (w, b) in zip([head[1], head[3], head[5]], [(w1, fc[0][1]), fc[1], fc[2]]):
            lin.weight.data = torch.from_numpy(np.ascontiguousarray(w.T)); lin.bias.data = torch.from_numpy(b)
        with torch.no_grad():
            p = head(backbone.features(prep_rgb(img, spec))).softmax(1)[0]
        print(f"VGGKNN      top-1 = {int(p.argmax())}  p(Samoyed) = {float(p[SAMOYED]):.3f}")

    if sq_top:
        spec = models.MODEL_SPECS["SqueezeKNN"]
        backbone = models.build_squeezenet11_from_keras(Path(sq_top))
        conv10 = nn.Conv2d(512, 1000, 1)
        with h5py.File(sq_top, "r") as h5:
            models._load_keras_conv(conv10, *models._keras_layer_params(h5, "conv10"))
        with torch.no_grad():
            p = torch.relu(conv10(backbone.features(prep_rgb(img, spec)))).mean((2, 3)).softmax(1)[0]
        print(f"SqueezeKNN  top-1 = {int(p.argmax())}  p(Samoyed) = {float(p[SAMOYED]):.3f}")


if __name__ == "__main__":
    a = sys.argv[1:] + [None, None]
    main(a[0], a[1], a[2])
