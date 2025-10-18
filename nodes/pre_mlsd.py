# nodes/pre_mlsd.py
import os
import numpy as np
from PIL import Image

import torch  # not strictly required but consistent with other nodes
from .utils import pil_list_from_bhwc, bhwc_from_pil_list

class PreMLSDWithLoaders:
    """
    MLSD (Mobile Line Segment Detection) preprocessor for ControlNet.
    - Tries ONNX MLSD if weights+onnxruntime are available.
    - Falls back to OpenCV Line Segment Detector (LSD) if not.
    Returns a 3-channel IMAGE (0..1) suitable for ControlNet.
    Passes through MODEL/CLIP/VAE/CONTROL_NET unchanged.
    """

    # Simple cache to avoid reloading backends/weights each call
    _ONNX_READY = False
    _ONNX_SESSION = None
    _LSD_READY = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "backend": (["auto", "onnx_mlsd", "opencv_lsd"],),
                # Rendering
                "line_thickness": ("INT", {"default": 1, "min": 1, "max": 9}),
                "render_style": (["white_on_black", "black_on_white", "grayscale"],),
                # Detection params
                "detect_resolution": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "min_line_length": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 1000.0}),
                "merge_distance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0}),
                # ONNX-only thresholds (ignored for OpenCV LSD)
                "mlsd_score_threshold": ("FLOAT", {"default": 0.10, "min": 0.01, "max": 1.0, "step": 0.01}),
                "mlsd_dist_threshold": ("FLOAT", {"default": 0.10, "min": 0.01, "max": 1.0, "step": 0.01}),
                # Weight search root (where .onnx files may live)
                "weights_dir": ("STRING", {"default": "weights/mlsd"}),
                "onnx_model": ("STRING", {"default": "mlsd_large_512_fp32.onnx"}),
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

    # ---------- utils ----------
    @staticmethod
    def _to_uint8_rgb(pil_img: Image.Image) -> np.ndarray:
        return np.asarray(pil_img.convert("RGB"), dtype=np.uint8)

    @staticmethod
    def _resize_keep_aspect(img: np.ndarray, target: int) -> np.ndarray:
        h, w = img.shape[:2]
        if max(h, w) == target:
            return img
        if h >= w:
            new_h = target
            new_w = int(round(w * (target / h)))
        else:
            new_w = target
            new_h = int(round(h * (target / w)))
        return np.array(Image.fromarray(img).resize((new_w, new_h), Image.BICUBIC))

    @staticmethod
    def _pad_to_square(img: np.ndarray, value: int = 0) -> np.ndarray:
        h, w = img.shape[:2]
        s = max(h, w)
        out = np.full((s, s, 3), value, dtype=np.uint8)
        y0 = (s - h) // 2
        x0 = (s - w) // 2
        out[y0:y0+h, x0:x0+w] = img
        return out, (x0, y0, w, h, s)

    @staticmethod
    def _crop_from_square(square: np.ndarray, meta) -> np.ndarray:
        x0, y0, w, h, s = meta
        return square[y0:y0+h, x0:x0+w]

    # ---------- ONNX MLSD ----------
    def _ensure_onnx(self, weights_dir: str, onnx_model: str):
        if self._ONNX_READY:
            return True

        try:
            import onnxruntime as ort
        except Exception:
            return False

        model_path = os.path.join(weights_dir, onnx_model)
        if not os.path.isfile(model_path):
            return False

        try:
            sess_opts = ort.SessionOptions()
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._ONNX_SESSION = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
            self._ONNX_READY = True
            return True
        except Exception:
            self._ONNX_READY = False
            self._ONNX_SESSION = None
            return False

    def _run_onnx_mlsd(self, rgb_u8: np.ndarray, score_th: float, dist_th: float, line_thickness: int,
                       detect_resolution: int, render_style: str) -> Image.Image:
        """
        Minimal ONNX MLSD implementation.
        Expects a 512x512 RGB float input in [0,1], NHWC → NCHW according to many public MLSD exports.
        Because MLSD ONNX variants differ, we include a best-effort path and fallback if shapes mismatch.
        """
        import onnxruntime as ort
        import cv2

        # Resize & pad to square 512x512 (as most MLSD weights are trained that way)
        img = self._resize_keep_aspect(rgb_u8, detect_resolution)
        sq, meta = self._pad_to_square(img, value=0)
        sq512 = np.array(Image.fromarray(sq).resize((512, 512), Image.BICUBIC), dtype=np.float32) / 255.0

        # NCHW
        x = sq512.transpose(2, 0, 1)[None, ...]  # (1,3,512,512)

        # Try to infer input / output names
        sess: ort.InferenceSession = self._ONNX_SESSION
        in_name = sess.get_inputs()[0].name
        outs = sess.run(None, {in_name: x})

        # Heuristic decode:
        # Some MLSD exports return line maps or junction maps. For portability,
        # we apply a simple postproc: detect prominent lines via probabilistic Hough on the output map,
        # falling back to drawing lines on a Canny of the input when model output is ambiguous.
        try:
            # Assume first output is a single-channel "line score" heatmap
            out0 = outs[0]
            if out0.ndim == 4:
                out0 = out0[0]  # (C,H,W)
            if out0.ndim == 3:
                if out0.shape[0] in (1, 2, 3, 4):  # pick strongest channel
                    heat = out0.max(axis=0)
                else:
                    # Unknown channel layout; reduce to mean
                    heat = out0.mean(axis=0)
            elif out0.ndim == 2:
                heat = out0
            else:
                raise RuntimeError("Unexpected MLSD output shape")

            heat = np.clip(heat, 0.0, None)
            heat = heat / (heat.max() + 1e-6)
            heat_u8 = (heat * 255).astype(np.uint8)

            # Threshold by score_th as a rough cutoff
            _, bw = cv2.threshold(heat_u8, int(score_th * 255), 255, cv2.THRESH_BINARY)

            # Find lines via Hough (coarse but robust across exports)
            lines = cv2.HoughLinesP(
                bw, rho=1, theta=np.pi / 180, threshold=max(10, int(dist_th * 100)),
                minLineLength=max(10, int(dist_th * 200)), maxLineGap=10
            )

            canvas = np.zeros_like(sq512, dtype=np.float32)
            if lines is not None:
                for l in lines[:, 0, :]:
                    x1, y1, x2, y2 = map(int, l)
                    cv2.line(canvas, (x1, y1), (x2, y2), (1.0, 1.0, 1.0), thickness=int(line_thickness))
            else:
                # Fallback: no lines found, just return thresholded heat as grayscale
                canvas = np.stack([bw, bw, bw], axis=-1).astype(np.float32) / 255.0

            # Un-pad back to original aspect
            canvas_u8 = (canvas * 255).astype(np.uint8)
            canvas_unpad = self._crop_from_square(canvas_u8, meta)
            canvas_unpad = np.array(Image.fromarray(canvas_unpad).resize((rgb_u8.shape[1], rgb_u8.shape[0]), Image.NEAREST))

        except Exception:
            # ONNX produced something unexpected → fallback to OpenCV LSD on input
            return self._run_opencv_lsd(rgb_u8, line_thickness, render_style, detect_resolution)

        return self._render(canvas_unpad, render_style)

    # ---------- OpenCV LSD ----------
    def _ensure_lsd(self):
        if self._LSD_READY:
            return True
        try:
            import cv2  # noqa
            self._LSD_READY = True
            return True
        except Exception:
            self._LSD_READY = False
            return False

    def _run_opencv_lsd(self, rgb_u8: np.ndarray, line_thickness: int, render_style: str, detect_resolution: int) -> Image.Image:
        import cv2

        # Work at detection resolution for stability
        det = self._resize_keep_aspect(rgb_u8, detect_resolution)
        gray = cv2.cvtColor(det, cv2.COLOR_RGB2GRAY)

        # Create LSD (LSD_REFINE_STD gives clean lines)
        try:
            lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        except Exception:
            lsd = cv2.createLineSegmentDetector()

        lines, _, _, _ = lsd.detect(gray)  # lines: Nx1x4

        # Prepare blank canvas 3ch
        canvas = np.zeros((det.shape[0], det.shape[1], 3), dtype=np.uint8)

        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), thickness=int(line_thickness))

        # Resize back to input size
        canvas = np.array(Image.fromarray(canvas).resize((rgb_u8.shape[1], rgb_u8.shape[0]), Image.NEAREST))
        return self._render(canvas, render_style)

    # ---------- Render helpers ----------
    @staticmethod
    def _render(img_u8_rgb: np.ndarray, render_style: str) -> Image.Image:
        """
        render_style:
          - white_on_black (default): as drawn (white lines on black)
          - black_on_white: invert
          - grayscale: keep as monochrome lines but in 3ch gray
        """
        out = img_u8_rgb
        if render_style == "black_on_white":
            out = 255 - out
        elif render_style == "grayscale":
            # already monochrome lines; convert to gray 3ch
            g = np.mean(out, axis=-1, keepdims=True).astype(np.uint8)
            out = np.repeat(g, 3, axis=-1)
        return Image.fromarray(out)

    # ---------- main ----------
    def run(
        self, image,
        backend,
        line_thickness,
        render_style,
        detect_resolution,
        min_line_length,
        merge_distance,
        mlsd_score_threshold,
        mlsd_dist_threshold,
        weights_dir,
        onnx_model,
        model=None, clip=None, vae=None, control_net=None
    ):
        pils = pil_list_from_bhwc(image)
        outs = []

        # Decide backend
        use_onnx = False
        if backend in ("auto", "onnx_mlsd"):
            use_onnx = self._ensure_onnx(weights_dir, onnx_model)

        # Ensure OpenCV availability if we might need it
        lsd_ok = self._ensure_lsd()

        for p in pils:
            rgb = self._to_uint8_rgb(p)

            if use_onnx:
                try:
                    out_pil = self._run_onnx_mlsd(
                        rgb, float(mlsd_score_threshold), float(mlsd_dist_threshold),
                        int(line_thickness), int(detect_resolution), render_style
                    )
                except Exception:
                    # Hard fallback to OpenCV LSD
                    out_pil = self._run_opencv_lsd(rgb, int(line_thickness), render_style, int(detect_resolution))
            else:
                if not lsd_ok:
                    raise RuntimeError(
                        "Neither ONNX MLSD (onnxruntime + weights) nor OpenCV LSD are available."
                    )
                out_pil = self._run_opencv_lsd(rgb, int(line_thickness), render_style, int(detect_resolution))

            outs.append(out_pil)

        control_image = bhwc_from_pil_list(outs)
        return (control_image, model, clip, vae, control_net)
