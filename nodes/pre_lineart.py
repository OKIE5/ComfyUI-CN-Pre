import numpy as np
from PIL import Image, ImageOps
from .utils import pil_list_from_bhwc, bhwc_from_pil_list, ensure_3ch_bhwc

try:
    from controlnet_aux.lineart import LineartDetector
    _lineart_available = True
except Exception as e:
    print(f"[PreLineArt] Warning: LineartDetector import failed: {e}")
    _lineart_available = False


class PreLineArt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "coarse": (["false", "true"],),
                "style": (["grayscale", "rgb"],),
                "invert": (["false", "true"],),
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "control_net": ("CONTROL_NET",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MODEL", "CLIP", "VAE", "CONTROL_NET")
    RETURN_NAMES = ("control_image", "model", "clip", "vae", "control_net")
    FUNCTION = "run"
    CATEGORY = "CtrlNet/Pre"

    def __init__(self):
        self.model = None

    def _lazy_init(self):
        if self.model is None:
            if not _lineart_available:
                raise ImportError("LineartDetector not available. Install with: pip install controlnet_aux")
            self.model = LineartDetector.from_pretrained("lllyasviel/Annotators")

    def run(
        self,
        image,
        coarse,
        style,
        invert,
        model=None,
        clip=None,
        vae=None,
        control_net=None,
    ):
        self._lazy_init()
        image = ensure_3ch_bhwc(image)
        pils = pil_list_from_bhwc(image)
        out_pils = []

        for p in pils:
            # Call the LineartDetector directly
            result = self.model(p)

            # Optional processing
            if style == "grayscale":
                result = ImageOps.grayscale(result)
            elif result.mode != "RGB":
                result = result.convert("RGB")

            if invert == "true":
                result = ImageOps.invert(result)

            out_pils.append(result)

        control_image = bhwc_from_pil_list(out_pils)
        return (control_image, model, clip, vae, control_net)
