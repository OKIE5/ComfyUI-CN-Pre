# nodes/pre_depth_depthanything.py
# Depth Anything preprocessor for ControlNet (V1 + V2).
# Robust V2: variant detection, constructor probing, and API shims.
# Keeps original class name/IO so existing __init__.py registration works.

import os
import sys
import numpy as np
import torch
from PIL import Image
from importlib import import_module
from .utils import pil_list_from_bhwc, bhwc_from_pil_list

# ---------------- path discovery (lazy import) ----------------
_THIS_FILE = os.path.abspath(__file__)
_NODES_DIR = os.path.dirname(_THIS_FILE)
_PLUGIN_DIR = os.path.abspath(os.path.join(_NODES_DIR, ".."))
_EXTERNALS_DIR = os.path.join(_PLUGIN_DIR, "externals")
_CUSTOM_NODES = os.path.abspath(os.path.join(_PLUGIN_DIR, ".."))

_ENV_HINTS = [
    os.environ.get("DEPTH_ANYTHING_V2_DIR", ""),
    os.environ.get("DEPTH_ANYTHING_DIR", ""),
]

_REPO_ROOTS = [
    os.path.join(_EXTERNALS_DIR, "Depth-Anything-V2"),
    os.path.join(_EXTERNALS_DIR, "depth_anything"),
    os.path.join(_CUSTOM_NODES, "Depth-Anything-V2"),
    os.path.join(_CUSTOM_NODES, "depth_anything"),
] + [p for p in _ENV_HINTS if p]

def _augment_sys_path_for_repos():
    candidates = []
    for root in _REPO_ROOTS:
        if not root:
            continue
        candidates.extend([
            root,
            os.path.join(root, "src"),
            os.path.join(root, "depth_anything"),
            os.path.join(root, "depth_anything_v2"),
            os.path.join(root, "src", "depth_anything"),
            os.path.join(root, "src", "depth_anything_v2"),
        ])
    seen = set()
    for p in candidates:
        if os.path.isdir(p) and p not in seen:
            seen.add(p)
            if p not in sys.path:
                sys.path.insert(0, p)

def _resolve_backend():
    """Return (module_name, class_name, api_tag, version). Prefer V2."""
    _augment_sys_path_for_repos()
    last_err = None
    try:
        mod = import_module("depth_anything_v2.dpt")
        if hasattr(mod, "DepthAnythingV2"):
            return ("depth_anything_v2.dpt", "DepthAnythingV2", "v2", 2)
    except Exception as e:
        last_err = e
    try:
        mod = import_module("depth_anything.dpt")
        if hasattr(mod, "DepthAnything"):
            return ("depth_anything.dpt", "DepthAnything", "v1", 1)
    except Exception as e:
        last_err = e
    roots = "\n  - " + "\n  - ".join(_REPO_ROOTS)
    raise ModuleNotFoundError(
        "[CN-PreDepthAnything] Could not locate Depth-Anything backend (V2 or V1).\n"
        "Tried 'depth_anything_v2.dpt' then 'depth_anything.dpt'.\n"
        f"Search roots:\n{roots}\n"
        "Set DEPTH_ANYTHING_V2_DIR or DEPTH_ANYTHING_DIR to the repo path.\n"
        f"Last import error: {last_err}"
    )

# ---------------- helpers ----------------

def _strip_prefix(state):
    """Accepts raw state, or {'state_dict': ...}; strips 'module.'/'model.'."""
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        return state
    out = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[7:]
        elif nk.startswith("model."):
            nk = nk[6:]
        out[nk] = v
    return out

def _embed_dim_from_state(sd: dict):
    """Read ViT embed dim from common keys."""
    keys = [
        "pretrained.patch_embed.proj.weight",
        "pretrained.pos_embed",
        "pretrained.cls_token",
        "encoder.patch_embed.proj.weight",
        "vit.patch_embed.proj.weight",
        "backbone.patch_embed.proj.weight",
    ]
    for k in keys:
        w = sd.get(k)
        if w is None:
            continue
        t = w if isinstance(w, torch.Tensor) else torch.as_tensor(w)
        if t.ndim >= 1:
            if k.endswith("proj.weight"):
                return int(t.shape[0])  # out_channels = embed dim
            return int(t.shape[-1])    # pos/cls last dim
    return None

def _infer_v2_variant(sd: dict):
    """Return one of: 'vits','vitb','vitl','vitg'."""
    dim = _embed_dim_from_state(sd)
    if dim in (384, 768, 1024, 1536):
        return {384: "vits", 768: "vitb", 1024: "vitl", 1536: "vitg"}[dim]
    # heuristic fallback from decoder head channels
    ch = None
    for k, w in sd.items():
        if isinstance(w, torch.Tensor) and w.ndim == 4:
            oc = int(w.shape[0])
            if oc in (64, 96, 128, 192, 256, 384, 768, 1024, 1536):
                ch = oc
                break
    if ch is None: return "vitb"
    if ch >= 1000 or ch == 384: return "vitg"
    if ch >= 200:               return "vitl"
    if ch >= 120:               return "vitb"
    return "vits"

def _enc_from_text(text: str):
    t = (text or "").lower()
    if "vitg" in t or "vit-g" in t: return "vitg"
    if "vitl" in t or "vit-l" in t: return "vitl"
    if "vitb" in t or "vit-b" in t: return "vitb"
    if "vits" in t or "vit-s" in t: return "vits"
    return None

def _vit_embed_mismatch(state: dict, net_state: dict) -> bool:
    """True if critical ViT tensors don’t match in shape."""
    for k in ("pretrained.cls_token", "pretrained.pos_embed", "pretrained.patch_embed.proj.weight"):
        if k in state and k in net_state:
            a, b = state[k], net_state[k]
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                if tuple(a.shape) != tuple(b.shape):
                    return True
    return False

def _construct_v2(Impl, enc: str):
    """
    Create a DepthAnythingV2 instance for a given encoder across common fork APIs.
    Tries: backbone=enc -> encoder=enc -> positional -> config-dict -> bare ().
    """
    # 1) backbone kw
    try:
        return Impl(backbone=enc)
    except TypeError:
        pass
    # 2) encoder kw
    try:
        return Impl(encoder=enc)
    except TypeError:
        pass
    # 3) positional first arg
    try:
        return Impl(enc)
    except TypeError:
        pass
    # 4) explicit config (older forks expect features/out_channels)
    cfgs = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    if enc in cfgs:
        try:
            return Impl(**cfgs[enc])
        except TypeError:
            pass
    # 5) last resort
    return Impl()

def _normalize(depth: np.ndarray, mode: str, fixed_min: float, fixed_max: float) -> np.ndarray:
    d = depth.astype(np.float32)
    if mode == "fixed_range":
        lo, hi = float(fixed_min), float(fixed_max)
        if abs(hi - lo) < 1e-6:
            hi = lo + 1e-6
        d = (d - lo) / (hi - lo)
    else:
        mn, mx = float(d.min()), float(d.max())
        if abs(mx - mn) < 1e-6:
            mx = mn + 1e-6
        d = (d - mn) / (mx - mn)
    return np.clip(d, 0.0, 1.0)

def _post_blur01(depth01: np.ndarray, kind: str, strength: float) -> np.ndarray:
    if kind == "none":
        return depth01
    try:
        import cv2
    except Exception:
        return depth01
    img = (depth01 * 255.0).astype(np.uint8)
    if kind == "gaussian":
        k = max(3, int(2 * round(strength) + 1))
        if k % 2 == 0: k += 1
        img = cv2.GaussianBlur(img, (k, k), strength if strength > 0 else 0)
    elif kind == "bilateral":
        sigma = max(1.0, strength * 20.0)
        img = cv2.bilateralFilter(img, d=9, sigmaColor=sigma, sigmaSpace=sigma)
    return img.astype(np.float32) / 255.0

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

def _call_v2_infer(net, pil_img: Image.Image):
    """
    Handle different V2 APIs:
      - infer_image(bgr_uint8)
      - infer(rgb_uint8)        (some forks)
      - predict(rgb_uint8)      (rare)
    """
    rgb = np.array(pil_img)  # HWC uint8 RGB
    if hasattr(net, "infer_image"):
        bgr = rgb[..., ::-1].copy()
        return net.infer_image(bgr)
    if hasattr(net, "infer"):
        return net.infer(rgb)
    if hasattr(net, "predict"):
        return net.predict(rgb)
    raise AttributeError("DepthAnythingV2 model has no infer_image/infer/predict method.")

def _call_v1_infer(net, pil_img: Image.Image):
    """
    Handle V1 APIs:
      - infer_pil(pil)
      - infer(pil) fallback
    """
    if hasattr(net, "infer_pil"):
        return net.infer_pil(pil_img)
    if hasattr(net, "infer"):
        return net.infer(pil_img)
    raise AttributeError("DepthAnything (V1) model has no infer_pil/infer method.")

# ---------------- node ----------------
class PreDepthAnythingWithLoaders:
    """
    Depth Anything preprocessor for ControlNet.
      • V2 (.pth via PTH Loader)  -> DepthAnythingV2(backbone/encoder/positional), then infer
      • V1 (HF model_id)          -> DepthAnything.from_pretrained(...), then infer
      • V1 (.pth via PTH Loader)  -> DepthAnything(arch=vit-*)
    Outputs: (IMAGE, MODEL, CLIP, VAE, CONTROL_NET)
    """
    _CACHE = {}  # ("v1", model_id) -> model

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_id": ("STRING", {"default": "Depth-Anything-V2-vit-b"}),
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

    # ---- V1 cache ----
    def _load_v1_from_pretrained(self, model_id: str, DepthAnything):
        key = ("v1", model_id)
        if key in self._CACHE:
            return self._CACHE[key]
        model = DepthAnything.from_pretrained(model_id)
        model.eval()
        self._CACHE[key] = model
        return model

    def run(
        self,
        image, model_id, normalize_mode, fixed_min, fixed_max, invert,
        post_blur, post_blur_strength, render_style,
        pth_model=None, model=None, clip=None, vae=None, control_net=None,
    ):
        mod, cls, api_tag, ver = _resolve_backend()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ---------- Build/Load ----------
        if pth_model is not None:
            Impl = getattr(import_module(mod), cls)
            state = _strip_prefix(pth_model)

            if ver == 2:
                # Determine target encoder
                hint = None
                try:
                    hint = getattr(pth_model, "__file__", None) or getattr(pth_model, "_origin", None)
                except Exception:
                    pass
                enc = _enc_from_text(hint or "") or _enc_from_text(model_id or "") or _infer_v2_variant(state)

                # Try detected enc first, then brute-try others until shapes match
                candidates = []
                if enc in ("vits", "vitb", "vitl", "vitg"):
                    candidates.append(enc)
                for e in ("vits", "vitb", "vitl", "vitg"):
                    if e not in candidates:
                        candidates.append(e)

                net = None
                errors = []
                for cand in candidates:
                    try:
                        tmp = _construct_v2(Impl, cand)
                        # Load with strict=False to tolerate naming, then verify embed shapes
                        _ = tmp.load_state_dict(state, strict=False)
                        if _vit_embed_mismatch(state, tmp.state_dict()):
                            errors.append((cand, "embed shape mismatch"))
                            continue
                        net = tmp.eval().to(device)
                        print(f"[CN-PreDepthAnything] V2 backbone selected: {cand}")
                        break
                    except Exception as e:
                        errors.append((cand, str(e)))
                        net = None

                if net is None:
                    raise RuntimeError(f"[CN-PreDepthAnything] Could not load V2 weights with any backbone. Tried: {errors}")

            else:
                # V1 .pth path
                name = (model_id or "").lower()
                if   ("vitg" in name) or ("vit-g" in name): arch = "vit-g"
                elif ("vitl" in name) or ("vit-l" in name): arch = "vit-l"
                elif ("vitb" in name) or ("vit-b" in name): arch = "vit-b"
                else: arch = "vit-s"
                net = Impl(arch=arch)
                _ = net.load_state_dict(state, strict=False)
                net = net.eval().to(device)

        else:
            # No .pth: only V1 supports from_pretrained() reliably
            if ver == 1:
                DepthAnything = getattr(import_module(mod), cls)
                net = self._load_v1_from_pretrained(model_id, DepthAnything).to(device)
            else:
                raise RuntimeError(
                    "[CN-PreDepthAnything] No .pth provided and V2 backend detected.\n"
                    "V2 has no from_pretrained(); load a V2 .pth via the PTH Loader."
                )

        # ---------- Inference ----------
        pils = pil_list_from_bhwc(image)
        outs = []
        with torch.no_grad():
            if ver == 1:
                for p in pils:
                    d = _call_v1_infer(net, p)
                    depth = d.detach().cpu().float().numpy() if isinstance(d, torch.Tensor) else d.astype(np.float32)
                    d01 = _normalize(depth, normalize_mode, fixed_min, fixed_max)
                    if invert == "true": d01 = 1.0 - d01
                    d01 = _post_blur01(d01, post_blur, float(post_blur_strength))
                    outs.append(_render(d01, render_style))
            else:
                for p in pils:
                    d = _call_v2_infer(net, p)   # accepts BGR or RGB per method
                    depth = d.detach().cpu().float().numpy() if isinstance(d, torch.Tensor) else d.astype(np.float32)
                    d01 = _normalize(depth, normalize_mode, fixed_min, fixed_max)
                    if invert == "true": d01 = 1.0 - d01
                    d01 = _post_blur01(d01, post_blur, float(post_blur_strength))
                    outs.append(_render(d01, render_style))

        control_image = bhwc_from_pil_list(outs)
        return (control_image, model, clip, vae, control_net)
