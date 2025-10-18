# nodes/pre_depth_midas_with_loaders.py
import numpy as np
import torch
from PIL import Image
from .utils import pil_list_from_bhwc, bhwc_from_pil_list

class PreDepthMiDaSWithLoaders:
    """
    MiDaS/DPT depth via torch.hub (downloads on first run).
    Returns 3ch IMAGE (0..1). Passes through MODEL/CLIP/VAE/CONTROL_NET.
    """
    _CACHE = {}  # model_name -> (model, transform)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (["DPT_Large", "DPT_Hybrid", "MiDaS_small"],),
                "normalize_mode": (["per_image_minmax", "fixed_range"],),
                "fixed_min": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "fixed_max": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "invert": (["false", "true"],),
                "post_blur": (["none", "gaussian", "bilateral"],),
                "post_blur_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0}),
                "render_style": (["grayscale", "pseudo_color"],),
            },
            "optional": {
                "pth_model": ("PTH_MODEL",),
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

    @staticmethod
    def _to_numpy01(t: torch.Tensor) -> np.ndarray:
        a = t.detach().cpu().float().numpy()
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        return a

    @staticmethod
    def _normalize(depth: np.ndarray, mode: str, fixed_min: float, fixed_max: float) -> np.ndarray:
        depth = depth.astype(np.float32)
        if mode == "fixed_range":
            lo, hi = float(fixed_min), float(fixed_max)
            if abs(hi - lo) < 1e-6: hi = lo + 1e-6
            d = (depth - lo) / (hi - lo)
        else:
            mn, mx = float(depth.min()), float(depth.max())
            if abs(mx - mn) < 1e-6: mx = mn + 1e-6
            d = (depth - mn) / (mx - mn)
        return np.clip(d, 0.0, 1.0)

    @staticmethod
    def _post_blur01(depth01: np.ndarray, kind: str, strength: float) -> np.ndarray:
        if kind == "none":
            return depth01
        try:
            import cv2
        except Exception:
            return depth01
        img = (depth01 * 255.0).astype(np.uint8)
        if kind == "gaussian":
            ksize = max(3, int(2 * round(strength) + 1)); 
            if ksize % 2 == 0: ksize += 1
            img = cv2.GaussianBlur(img, (ksize, ksize), strength if strength > 0 else 0)
        elif kind == "bilateral":
            sigma = max(1.0, strength * 20.0)
            img = cv2.bilateralFilter(img, d=9, sigmaColor=sigma, sigmaSpace=sigma)
        return img.astype(np.float32) / 255.0

    @staticmethod
    def _render(depth01: np.ndarray, style: str) -> Image.Image:
        img = (depth01 * 255.0).astype(np.uint8)
        if style == "pseudo_color":
            try:
                import cv2
                col = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)[..., ::-1]
                return Image.fromarray(col)
            except Exception:
                pass
        return Image.fromarray(np.stack([img, img, img], axis=-1))

    def _load_midas(self, model_name: str):
        if model_name in self._CACHE:
            return self._CACHE[model_name]
        try:
            import torch.hub
            repo = "intel-isl/MiDaS"
            model = torch.hub.load(repo, model_name)
            model.eval()
            transforms = torch.hub.load(repo, "transforms")
            tfm = transforms.dpt_transform if "DPT" in model_name else transforms.small_transform
            self._CACHE[model_name] = (model, tfm)
            return self._CACHE[model_name]
        except Exception as e:
            raise RuntimeError(
                f"MiDaS failed to load '{model_name}'. Ensure internet on first run or cached weights. Details: {e}"
            )

    def run(
        self, image, model_name, normalize_mode, fixed_min, fixed_max, invert,
        post_blur, post_blur_strength, render_style, model=None, clip=None, vae=None, control_net=None
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net, tfm = self._load_midas(model_name)
        net.to(device)

        pils = pil_list_from_bhwc(image)
        outs = []

        with torch.no_grad():
            for p in pils:
                # >>> FIX: MiDaS transforms expect a NumPy array, not PIL <<<
                arr = np.asarray(p.convert("RGB"))  # HxWx3 uint8
                transformed = tfm(arr)              # dict or tensor/ndarray

                # unwrap dict{'image': ...}
                if isinstance(transformed, dict):
                    x = transformed.get("image", transformed)
                else:
                    x = transformed

                # ensure torch tensor with batch dim
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x)
                if x.ndim == 3:               # [3,H,W] -> [1,3,H,W]
                    x = x.unsqueeze(0)
                elif x.ndim != 4:
                    raise RuntimeError(f"Unexpected MiDaS transform output shape: {tuple(x.shape)}")

                x = x.to(device)
                pred = net(x)  # [1,1,H,W] or [1,H,W]
                depth = self._to_numpy01(pred.squeeze())

                d01 = self._normalize(depth, normalize_mode, fixed_min, fixed_max)
                if invert == "true":
                    d01 = 1.0 - d01
                d01 = self._post_blur01(d01, post_blur, float(post_blur_strength))
                outs.append(self._render(d01, render_style))

        control_image = bhwc_from_pil_list(outs)
        return (control_image, model, clip, vae, control_net)
