from .utils import resize_bhwc, nearest_multiple_of_8

class ResizeForControl:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "image": ("IMAGE",),
            "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
            "width":  ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
            "snap_to_x8": (["true","false"],),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("control_image",)
    FUNCTION = "run"
    CATEGORY = "CtrlNet/Pre"

    def run(self, image, height, width, snap_to_x8):
        if snap_to_x8 == "true":
            height = nearest_multiple_of_8(height)
            width  = nearest_multiple_of_8(width)
        return (resize_bhwc(image, height, width),)
