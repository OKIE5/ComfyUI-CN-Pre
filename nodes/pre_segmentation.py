# nodes/pre_segmentation.py
import os
import numpy as np
from PIL import Image

import torch
from .utils import pil_list_from_bhwc, bhwc_from_pil_list

# ---- ADE20K palette (150) ----
_ADE20K_PALETTE = [
    (0,0,0),(120,120,120),(180,120,120),(6,230,230),(80,50,50),
    (4,200,3),(120,120,80),(140,140,140),(204,5,255),(230,230,230),
    (4,250,7),(224,5,255),(235,255,7),(150,5,61),(120,120,70),
    (8,255,51),(255,6,82),(143,255,140),(204,255,4),(255,51,7),
    (204,70,3),(0,102,200),(61,230,250),(255,6,51),(11,102,255),
    (255,7,71),(255,9,224),(9,7,230),(220,220,220),(255,9,92),
    (112,9,255),(8,255,214),(7,255,224),(255,184,6),(10,255,71),
    (255,41,10),(7,255,255),(224,255,8),(102,8,255),(255,61,6),
    (255,194,7),(255,122,8),(0,255,20),(255,8,41),(255,5,153),
    (6,51,255),(235,12,255),(160,150,20),(0,163,255),(140,140,140),
    (250,10,15),(20,255,0),(31,255,0),(255,31,0),(255,224,0),
    (153,255,0),(0,0,255),(255,71,0),(0,235,255),(0,173,255),
    (31,0,255),(11,200,200),(255,82,0),(0,255,245),(0,61,255),
    (0,255,112),(0,255,133),(255,0,0),(255,163,0),(255,102,0),
    (194,255,0),(0,143,255),(51,255,0),(0,82,255),(0,255,41),
    (0,255,173),(10,0,255),(173,255,0),(0,255,153),(255,92,0),
    (255,0,255),(255,0,245),(255,0,102),(255,173,0),(255,0,20),
    (255,184,184),(0,31,255),(0,255,61),(0,71,255),(255,0,204),
    (0,255,194),(0,255,82),(0,10,255),(0,112,255),(51,0,255),
    (0,194,255),(0,122,255),(0,255,163),(255,153,0),(0,255,10),
    (255,112,0),(143,255,0),(82,0,255),(163,255,0),(255,235,0),
    (8,184,170),(133,0,255),(0,255,92),(184,0,255),(255,0,31),
    (0,184,255),(0,214,255),(255,0,112),(92,255,0),(0,224,255),
    (112,224,255),(70,184,160),(163,0,255),(153,0,255),(71,255,0),
    (255,0,163),(255,204,0),(255,0,143),(0,255,235),(133,255,0),
    (255,0,235),(245,0,255),(255,0,122),(255,245,0),(10,190,212),
    (214,255,0),(0,204,255),(20,0,255),(255,255,0),(0,153,255),
    (0,41,255),(0,255,204),(41,0,255),(41,255,0),(173,0,255),
    (0,245,255),(71,0,255),(122,0,255),(0,255,184),(0,92,255),
    (184,255,0),(0,133,255),(255,214,0),(25,194,194),(102,255,0),
    (92,0,255)
]
_PALETTE = np.array(_ADE20K_PALETTE, dtype=np.uint8)


class PreSegmentationWithLoaders:
    """
    Semantic segmentation preprocessor for ControlNet (Seg).
    Backends:
      - ONNX SegFormer (ADE20K) if onnxruntime + weights are present.
      - TorchVision DeepLabV3 (COCO) fallback with multi-scale logits fusion.
    Output: 3ch IMAGE [0..1] (palette- or gray-colored) + passthrough MODEL/CLIP/VAE/CONTROL_NET.
    """

    _ONNX_READY = False
    _ONNX_SESSION = None
    _TORCH_DEEPLAB_READY = False
    _DEEPLAB = None
    _DL_TRANSFORMS = None  # official preprocessing from weights

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "backend": (["auto", "onnx_segformer", "torch_deeplab"],),
                "render_style": (["ade20k_palette", "random_palette", "class_index_gray"],),
                "detect_resolution": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "weights_dir": ("STRING", {"default": "weights/segmentation"}),
                "onnx_model": ("STRING", {"default": "segformer_b5_ade20k_640x640.onnx"}),
                "background_color": (["dark_gray", "white", "black"],),
                "edge_overlay": (["off", "on"],),
            },
            "optional": {
                # passthroughs to fit "+Loaders" chain
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "control_net": ("CONTROL_NET",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MODEL", "CLIP", "VAE", "CONTROL_NET")
    RETURN_NAMES = ("control_image", "model", "clip", "vae", "control_net")
    FUNCTION = "run"
    CATEGORY = "CtrlNet/Pre"

    # ---------- helpers ----------
    @staticmethod
    def _pil_to_u8rgb(pil: Image.Image) -> np.ndarray:
        return np.asarray(pil.convert("RGB"), dtype=np.uint8)

    @staticmethod
    def _resize_keep_aspect(img: np.ndarray, target: int) -> np.ndarray:
        h, w = img.shape[:2]
        if max(h, w) == target:
            return img
        if h >= w:
            nh, nw = target, int(round(w * (target / h)))
        else:
            nw, nh = target, int(round(h * (target / w)))
        return np.array(Image.fromarray(img).resize((nw, nh), Image.BICUBIC))

    @staticmethod
    def _pad_to_square(img: np.ndarray, value: int = 0):
        h, w = img.shape[:2]
        s = max(h, w)
        out = np.full((s, s, 3), value, dtype=np.uint8)
        y0 = (s - h) // 2
        x0 = (s - w) // 2
        out[y0:y0 + h, x0:x0 + w] = img
        return out, (x0, y0, w, h, s)

    @staticmethod
    def _crop_from_square(square: np.ndarray, meta):
        x0, y0, w, h, s = meta
        return square[y0:y0 + h, x0:x0 + w]

    @staticmethod
    def _label_edges(labels: np.ndarray) -> np.ndarray:
        h, w = labels.shape
        edges = np.zeros_like(labels, dtype=np.uint8)
        edges[1:, :] |= (labels[1:, :] != labels[:-1, :])
        edges[:-1, :] |= (labels[:-1, :] != labels[1:, :])
        edges[:, 1:] |= (labels[:, 1:] != labels[:, :-1])
        edges[:, :-1] |= (labels[:, :-1] != labels[:, 1:])
        return edges * 255

    @staticmethod
    def _bg_color_rgb(name: str) -> np.ndarray:
        if name == "white":
            return np.array([240, 240, 240], dtype=np.uint8)
        if name == "black":
            return np.array([0, 0, 0], dtype=np.uint8)
        return np.array([40, 40, 40], dtype=np.uint8)  # dark_gray

    def _colorize_labels(self, labels_hw: np.ndarray, style: str, background_color: str, edge_overlay: bool) -> np.ndarray:
        h, w = labels_hw.shape
        bg_ratio = (labels_hw == 0).mean()
        almost_all_bg = bg_ratio > 0.95

        if style == "class_index_gray":
            g = (labels_hw.astype(np.float32) / (labels_hw.max() + 1e-6)) * 255.0
            g = g.astype(np.uint8)
            color = np.stack([g, g, g], axis=-1)
        else:
            if style == "random_palette" or almost_all_bg:
                rng = np.random.default_rng(12345)
                max_lab = int(labels_hw.max()) + 1
                pal = rng.integers(0, 256, size=(max(151, max_lab + 1), 3), dtype=np.uint8)
            else:
                pal = _PALETTE.copy()
            pal[0] = self._bg_color_rgb(background_color)
            idx = labels_hw % pal.shape[0]
            color = pal[idx]

        if edge_overlay:
            edges = self._label_edges(labels_hw)
            draw_col = np.array([0, 0, 0], dtype=np.uint8) if (background_color != "black") else np.array([255, 255, 255], dtype=np.uint8)
            mask = edges.astype(bool)
            color[mask] = draw_col

        return color

    # ---------- ONNX SegFormer (ADE20K) ----------
    def _ensure_onnx(self, weights_dir: str, onnx_model: str):
        if self._ONNX_READY:
            return True
        try:
            import onnxruntime as ort  # noqa
        except Exception:
            return False

        model_path = os.path.join(weights_dir, onnx_model)
        if not os.path.isfile(model_path):
            return False

        try:
            import onnxruntime as ort
            sess_opts = ort.SessionOptions()
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._ONNX_SESSION = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
            self._ONNX_READY = True
            return True
        except Exception:
            self._ONNX_SESSION = None
            self._ONNX_READY = False
            return False

    def _run_onnx_segformer(self, rgb_u8: np.ndarray, detect_resolution: int, render_style: str, background_color: str, edge_overlay: bool) -> Image.Image:
        img = self._resize_keep_aspect(rgb_u8, detect_resolution)
        sq, meta = self._pad_to_square(img, value=0)

        target = 640
        sqT = np.array(Image.fromarray(sq).resize((target, target), Image.BICUBIC), dtype=np.float32) / 255.0
        x = sqT.transpose(2, 0, 1)[None, ...].astype(np.float32)

        sess = self._ONNX_SESSION
        in_name = sess.get_inputs()[0].name
        out = sess.run(None, {in_name: x})[0]
        if out.ndim == 4:
            out = out[0]

        out = out - out.max(axis=0, keepdims=True)
        exp = np.exp(out)
        prob = exp / (exp.sum(axis=0, keepdims=True) + 1e-12)
        labels = np.argmax(prob, axis=0).astype(np.int32)

        labels_hw = np.array(
            Image.fromarray(labels.astype(np.int32), mode="I").resize((sq.shape[1], sq.shape[0]), Image.NEAREST)
        )
        labels_unpad = self._crop_from_square(labels_hw, meta)
        color = self._colorize_labels(labels_unpad, render_style, background_color, edge_overlay)
        color = np.array(Image.fromarray(color).resize((rgb_u8.shape[1], rgb_u8.shape[0]), Image.NEAREST))
        return Image.fromarray(color)

    # ---------- TorchVision DeepLabV3 (COCO) with multi-scale fusion ----------
    def _ensure_deeplab(self):
        if self._TORCH_DEEPLAB_READY:
            return True
        try:
            import torchvision
            from torchvision.models.segmentation import deeplabv3_resnet50
            from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
        except Exception:
            return False

        try:
            weights = DeepLabV3_ResNet50_Weights.DEFAULT
            self._DEEPLAB = deeplabv3_resnet50(weights=weights).eval()
            self._DL_TRANSFORMS = weights.transforms()
        except Exception:
            self._DEEPLAB = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True).eval()
            import torchvision.transforms as T
            self._DL_TRANSFORMS = T.Compose([T.ToTensor()])

        self._TORCH_DEEPLAB_READY = True
        return True

    @torch.no_grad()
    def _deeplab_logits_official(self, pil_img: Image.Image, device: str):
        net = self._DEEPLAB.to(device)
        x = self._DL_TRANSFORMS(pil_img).unsqueeze(0).to(device)  # [1,3,H,W]
        out = net(x)["out"]  # [1,C,H,W]
        return out  # logits

    @torch.no_grad()
    def _deeplab_logits_hires(self, pil_img: Image.Image, device: str):
        # Manual 960 long-side + ImageNet normalization
        import torchvision.transforms as T
        w, h = pil_img.size
        if w >= h:
            new_w, new_h = 960, int(round(h * (960 / max(w, 1))))
        else:
            new_h, new_w = 960, int(round(w * (960 / max(h, 1))))
        resized = pil_img.resize((new_w, new_h), Image.BICUBIC)

        tfm = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        x = tfm(resized).unsqueeze(0).to(device)  # [1,3,h',w']
        net = self._DEEPLAB.to(device)
        out = net(x)["out"]  # [1,C,h',w']

        # Upsample logits back to original size
        out = torch.nn.functional.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out  # logits aligned to input size

    @torch.no_grad()
    def _run_torch_deeplab(self, rgb_u8: np.ndarray, render_style: str, background_color: str, edge_overlay: bool) -> Image.Image:
        pil_img = Image.fromarray(rgb_u8)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1) official-transform pass
        logits1 = self._deeplab_logits_official(pil_img, device)  # [1,C,H1,W1]
        # upsample to original so sizes match for fusion
        h, w = pil_img.size[1], pil_img.size[0]
        logits1 = torch.nn.functional.interpolate(logits1, size=(h, w), mode="bilinear", align_corners=False)

        # 2) hi-res pass (long side 960)
        logits2 = self._deeplab_logits_hires(pil_img, device)

        # Fuse (average) logits, then argmax
        logits = (logits1 + logits2) / 2.0
        labels = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.int32)

        # If degenerate, force an edge overlay so it’s not a flat panel
        if int(np.unique(labels).size) < 2:
            edge_overlay = True

        color = self._colorize_labels(labels, render_style, background_color, edge_overlay)
        return Image.fromarray(color)

    # ---------- main ----------
    def run(
        self, image,
        backend, render_style, detect_resolution,
        weights_dir, onnx_model,
        background_color, edge_overlay,
        model=None, clip=None, vae=None, control_net=None
    ):
        pils = pil_list_from_bhwc(image)
        outs = []

        # backend selection
        use_onnx = False
        if backend in ("auto", "onnx_segformer"):
            use_onnx = self._ensure_onnx(weights_dir, onnx_model)

        use_torch = False
        if not use_onnx and backend in ("auto", "torch_deeplab"):
            use_torch = self._ensure_deeplab()

        if not use_onnx and not use_torch:
            raise RuntimeError(
                "No segmentation backend available. "
                "Provide ONNX SegFormer (onnxruntime + weights) or install torchvision for DeepLabV3."
            )

        edge_flag = (edge_overlay == "on")

        for p in pils:
            rgb = self._pil_to_u8rgb(p)
            if use_onnx:
                try:
                    out_pil = self._run_onnx_segformer(rgb, int(detect_resolution), render_style, background_color, edge_flag)
                except Exception:
                    if use_torch or self._ensure_deeplab():
                        out_pil = self._run_torch_deeplab(rgb, render_style, background_color, edge_flag)
                    else:
                        raise
            else:
                out_pil = self._run_torch_deeplab(rgb, render_style, background_color, edge_flag)
            outs.append(out_pil)

        control_image = bhwc_from_pil_list(outs)
        return (control_image, model, clip, vae, control_net)


# ---- Export mappings so your loader can merge them ----
NODE_CLASS_MAPPINGS = {
    "CN Pre+Loaders: Segmentation": PreSegmentationWithLoaders,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CN Pre+Loaders: Segmentation": "ControlNet Pre (+Model/ControlNet): Segmentation",
}
