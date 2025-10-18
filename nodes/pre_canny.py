import numpy as np
from PIL import ImageOps, Image
from .utils import pil_list_from_bhwc, bhwc_from_pil_list, ensure_3ch_bhwc

class PreCannyWithLoaders:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "low_threshold": ("INT", {"default": 100, "min": 0, "max": 255}),
                "high_threshold": ("INT", {"default": 200, "min": 0, "max": 255}),
                # OpenCV Canny supports Sobel aperture sizes of 3, 5, or 7
                "aperture_size": (["3", "5", "7"],),
                "blur_type": (["none", "gaussian", "median", "bilateral"],),
                "blur_radius": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0}),
                "dilate_iter": ("INT", {"default": 0, "min": 0, "max": 5}),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 7}),
                "invert_output": (["false", "true"],),
                "edge_gain": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0}),
                "edge_bias": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5}),
            },
            "optional": {
                # Pass-throughs from CheckpointLoaderSimple & ControlNetLoader
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

    def run(
        self,
        image,
        low_threshold,
        high_threshold,
        aperture_size,
        blur_type,
        blur_radius,
        dilate_iter,
        kernel_size,
        invert_output,
        edge_gain,
        edge_bias,
        model=None,
        clip=None,
        vae=None,
        control_net=None,
    ):
        """
        Returns a 3-channel edge map as IMAGE, plus passes through model/clip/vae/control_net.
        """
        image = ensure_3ch_bhwc(image)
        pils = pil_list_from_bhwc(image)
        out_pils = []

        # sanitize params
        try:
            ap_size = int(aperture_size)
        except Exception:
            ap_size = 3
        if ap_size not in (3, 5, 7):
            ap_size = 3

        k = max(1, int(kernel_size))
        if k % 2 == 0:
            k += 1  # force odd for morphology/median
        br = float(blur_radius)

        try:
            import cv2
            use_cv2 = True
        except Exception:
            use_cv2 = False

        for p in pils:
            arr_gray = np.asarray(ImageOps.grayscale(p))

            # ---- pre-blur (noise reduction) ----
            if use_cv2 and blur_type != "none":
                if blur_type == "gaussian":
                    # (ksize auto from radius: ensure odd and >=3)
                    ksize = max(3, int(2 * round(br) + 1))
                    if ksize % 2 == 0:
                        ksize += 1
                    arr_gray = cv2.GaussianBlur(arr_gray, (ksize, ksize), br if br > 0 else 0)
                elif blur_type == "median":
                    ksize = max(3, int(2 * round(br) + 1))
                    if ksize % 2 == 0:
                        ksize += 1
                    arr_gray = cv2.medianBlur(arr_gray, ksize)
                elif blur_type == "bilateral":
                    # diameter 9 with sigma scaled by radius
                    sigma = max(1.0, br * 20.0)
                    arr_gray = cv2.bilateralFilter(arr_gray, d=9, sigmaColor=sigma, sigmaSpace=sigma)

            # ---- canny ----
            if use_cv2:
                edges = cv2.Canny(arr_gray, int(low_threshold), int(high_threshold), apertureSize=ap_size)
                # dilation (thicken edges)
                if dilate_iter > 0:
                    kernel = np.ones((k, k), np.uint8)
                    edges = cv2.dilate(edges, kernel, iterations=int(dilate_iter))
            else:
                # Fallback Sobel-magnitude if cv2 missing
                g = arr_gray.astype(np.float32)
                gx = np.zeros_like(g); gy = np.zeros_like(g)
                gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
                gy[1:-1, :] = g[2:, :] - g[:-2, :]
                edges = np.clip(np.hypot(gx, gy), 0, 255).astype(np.uint8)

            # ---- output post-adjust ----
            # gain/bias on 0..255 domain
            edges = edges.astype(np.float32) * float(edge_gain) + float(edge_bias) * 255.0
            edges = np.clip(edges, 0, 255).astype(np.uint8)

            if invert_output == "true":
                edges = 255 - edges

            rgb = np.stack([edges] * 3, axis=-1)
            out_pils.append(Image.fromarray(rgb))

        control_image = bhwc_from_pil_list(out_pils)
        return (control_image, model, clip, vae, control_net)
