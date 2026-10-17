r"""
braintumor.py - the tumor-type CLASSIFIER trained on Colab (Brain Tumor MRI dataset, Nickparvar).

what it is:   the ResNet34 encoder + classification head of a U-Net (smp.Unet). It names the tumor type:
              glioma / meningioma / no tumor / pituitary.
what it is NOT: it does NOT draw the tumor region. That dataset has no masks. The "Grad-CAM" picture only shows
              where the model looked. It is not a tumor outline.

files it needs (from the Colab zip braintumor_results.zip):
    models\braintumor_unet_resnet34_cls.pt              <- the model
    outputs\braintumor\test_results.json, training_log.csv, confusion_matrix.png   <- for the Benchmarks page (optional)

test it alone:   py braintumor.py predict "C:\path\to\picture.jpg"
"""
import math
import sys
import time
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    import segmentation_models_pytorch as smp
    from PIL import Image
    import medical_assistant as ma
except ImportError as import_error:
    sys.exit(f"a library is missing: {import_error.name}\ninstall:  python -m pip install segmentation-models-pytorch pillow numpy")

MODEL_NAME = "braintumor_unet_resnet34_cls.pt"
DISPLAY = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "No tumor", "pituitary": "Pituitary tumor"}
LOW_CONFIDENCE = 0.60         # under this the app says "the model is not sure"
MEAN = (0.485, 0.456, 0.406)  # ImageNet numbers (the same as in the Colab notebook)
STD = (0.229, 0.224, 0.225)


def paths():
    tag = "_demo" if ma.DEMO else ""
    return type("P", (), {"model": ma.PROJECT_DIR / "models" / MODEL_NAME,
                          "out": ma.PROJECT_DIR / "outputs" / f"braintumor{tag}"})()


class TumorClassifier:
    def __init__(self, checkpoint_path=None, device=None):
        path = Path(checkpoint_path or paths().model)
        if not path.exists():
            raise FileNotFoundError(f"classifier file not found: {path}\ncopy braintumor_unet_resnet34_cls.pt from the Colab zip into the models folder")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(path, map_location=self.device)
        self.classes = list(ckpt["classes"])
        self.meta = {"image_size": int(ckpt["img_size"]), "epoch": int(ckpt["epoch"]),
                     "val_f1": float(ckpt["val_f1"]), "val_acc": float(ckpt["val_acc"])}
        self.model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1,
                              aux_params={"classes": len(self.classes), "pooling": "avg", "dropout": 0.3})
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1)

    def _logits(self, x01):
        x = (x01 - self.mean) / self.std
        return self.model.classification_head(self.model.encoder(x)[-1])      # encoder -> head (the decoder is not used)

    def warmup(self):
        size = self.meta["image_size"]
        with torch.no_grad():
            self._logits(torch.zeros(1, 3, size, size, device=self.device))

    def analyze(self, image, with_cam=True):
        """image: a PIL image. returns label, display, confidence, probs, uncertain, cam (0..1, size x size), rgb, seconds."""
        start = time.perf_counter()
        size = self.meta["image_size"]
        rgb = np.array(image.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)
        x01 = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255

        store = {}
        layer = self.model.encoder.layer4
        handles = []
        if with_cam:
            handles = [layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o)),
                       layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("g", go[0]))]
        try:
            with torch.enable_grad() if with_cam else torch.no_grad():
                x = x01.clone().requires_grad_(with_cam)
                logits = self._logits(x)
                probs = torch.softmax(logits.detach(), dim=1)[0].cpu().numpy()
                best = int(probs.argmax())
                cam = None
                if with_cam:
                    self.model.zero_grad()
                    logits[0, best].backward()
                    weights = store["g"].mean(dim=(2, 3), keepdim=True)
                    cam = F.relu((weights * store["a"]).sum(1, keepdim=True))
                    cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
                    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        finally:
            for handle in handles:
                handle.remove()
        label = self.classes[best]
        return {"label": label, "display": DISPLAY.get(label, label), "confidence": float(probs[best]),
                "probs": {c: float(p) for c, p in zip(self.classes, probs)},
                "uncertain": float(probs[best]) < LOW_CONFIDENCE,
                "cam": cam, "rgb": rgb, "seconds": round(time.perf_counter() - start, 3)}


def cam_overlay(rgb, cam, alpha=0.45):
    """a PIL picture: the scan with the Grad-CAM heat map on top (blue = ignored, red = looked here)."""
    t = np.clip(cam, 0, 1)
    heat = np.stack([np.clip(1.5 - np.abs(4 * t - 3), 0, 1), np.clip(1.5 - np.abs(4 * t - 2), 0, 1),
                     np.clip(1.5 - np.abs(4 * t - 1), 0, 1)], axis=-1)           # a simple jet-like color map
    mixed = (1 - alpha) * rgb.astype(np.float32) / 255 + alpha * heat
    return Image.fromarray((np.clip(mixed, 0, 1) * 255).astype(np.uint8))


def main():
    if len(sys.argv) < 3 or sys.argv[1] != "predict":
        sys.exit('use:  py braintumor.py predict "C:\\path\\to\\picture.jpg"')
    image_path = Path(sys.argv[2])
    if not image_path.exists():
        sys.exit(f"file not found: {image_path}")
    classifier = TumorClassifier()
    result = classifier.analyze(Image.open(image_path))
    print(f"\nprediction: {result['display']}  ({100 * result['confidence']:.1f}%)" + ("   <- the model is not sure" if result["uncertain"] else ""))
    for label, p in result["probs"].items():
        print(f"  {DISPLAY.get(label, label):16s} {100 * p:5.1f}%")
    out = image_path.with_name(image_path.stem + "_gradcam.png")
    cam_overlay(result["rgb"], result["cam"]).save(out)
    print(f"\nGrad-CAM picture saved: {out}   (shows where the model looked. NOT a tumor outline)")


if __name__ == "__main__":
    main()
