import torch
from .utils import ensure_3ch_bhwc

class ControlNormalize:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "image": ("IMAGE",),
            "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05}),
            "gain":  ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
            "bias":  ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            "invert": (["false","true"],),
            "clip":  (["true","false"],),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("control_image",)
    FUNCTION = "run"
    CATEGORY = "CtrlNet/Pre"

    def run(self, image, gamma, gain, bias, invert, clip):
        x = ensure_3ch_bhwc(image).float()
        if invert == "true": x = 1 - x
        x = torch.pow(x.clamp(0,1), 1.0/max(gamma,1e-6))
        x = x * gain + bias
        if clip == "true": x = x.clamp(0,1)
        return (x.contiguous(),)
