# nodes/levels.py
import numpy as np
from PIL import Image
from .utils import pil_list_from_bhwc, bhwc_from_pil_list

class ControlLevels:
    """
    Photoshop-like Levels for ComfyUI (image-only):
      - black_point: input level that maps to 0
      - white_point: input level that maps to 1
      - gamma: midtone (center balance), >1 darkens mids, <1 brightens mids

    Inputs:  IMAGE (HxWx3, values in [0..1])
    Outputs: IMAGE (HxWx3, values in [0..1])

    Optional: per-channel mode with independent R/G/B black/white/gamma.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "black_point": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "white_point": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.10, "max": 5.0, "step": 0.01}),
                "per_channel": (["false", "true"],),
                "bp_r": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "wp_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "gm_r": ("FLOAT", {"default": 1.0, "min": 0.10, "max": 5.0, "step": 0.01}),
                "bp_g": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "wp_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "gm_g": ("FLOAT", {"default": 1.0, "min": 0.10, "max": 5.0, "step": 0.01}),
                "bp_b": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "wp_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "gm_b": ("FLOAT", {"default": 1.0, "min": 0.10, "max": 5.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "Image/Adjust"

    @staticmethod
    def _levels_core(arr01: np.ndarray, bp: float, wp: float, gm: float) -> np.ndarray:
        # Guard against degenerate ranges
        if wp <= bp:
            wp = bp + 1e-6
        # Normalize to 0..1
        out = (arr01 - bp) / (wp - bp)
        out = np.clip(out, 0.0, 1.0)
        # Gamma correction (Photoshop Levels uses inverse exponent)
        out = out ** (1.0 / max(gm, 1e-6))
        return np.clip(out, 0.0, 1.0)

    def run(
        self,
        image,
        black_point, white_point, gamma,
        per_channel,
        bp_r, wp_r, gm_r, bp_g, wp_g, gm_g, bp_b, wp_b, gm_b,
    ):
        pils = pil_list_from_bhwc(image)
        outs = []

        per_ch = (per_channel == "true")

        for p in pils:
            rgb = np.asarray(p.convert("RGB"), dtype=np.float32) / 255.0  # HxWx3 in [0..1]

            if per_ch:
                r = self._levels_core(rgb[..., 0], float(bp_r), float(wp_r), float(gm_r))
                g = self._levels_core(rgb[..., 1], float(bp_g), float(wp_g), float(gm_g))
                b = self._levels_core(rgb[..., 2], float(bp_b), float(wp_b), float(gm_b))
                out = np.stack([r, g, b], axis=-1)
            else:
                out = self._levels_core(rgb, float(black_point), float(white_point), float(gamma))

            outs.append(Image.fromarray((np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)))

        image_out = bhwc_from_pil_list(outs)
        return (image_out,)


# Auto-register for your loader
NODE_CLASS_MAPPINGS = {
    "Image: Levels": ControlLevels,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Image: Levels": "Levels (Black/White/Gamma)",
}
