# nodes/pth_loader.py
import os
import torch

class PTHLoader:
    """
    Simple .pth / .pt model loader.
    Auto-scans ComfyUI 'models' folder for files and presents a dropdown.
    Outputs a loaded torch object or state_dict for use in custom preprocessors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Scan common ComfyUI model directories for .pth / .pt
        base_dir = os.path.join(os.getcwd(), "models")
        pth_files = []
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.lower().endswith((".pth", ".pt")):
                    rel_path = os.path.relpath(os.path.join(root, f), base_dir)
                    pth_files.append(rel_path)
        if not pth_files:
            pth_files = ["<no .pth files found>"]

        return {
            "required": {
                "model_file": (pth_files,),
                "map_location": (["auto", "cpu", "cuda"],),
            }
        }

    RETURN_TYPES = ("PTH_MODEL",)
    RETURN_NAMES = ("pth_model",)
    FUNCTION = "load"
    CATEGORY = "CtrlNet/Pre"

    _CACHE = {}

    def _resolve_device(self, map_location):
        if map_location == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return map_location

    def load(self, model_file, map_location):
        base_dir = os.path.join(os.getcwd(), "models")
        full_path = os.path.join(base_dir, model_file)

        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"[ComfyUI-PTHLoader] File not found: {full_path}")

        device = self._resolve_device(map_location)
        cache_key = (full_path, device)

        if cache_key in self._CACHE:
            print(f"[ComfyUI-PTHLoader] Using cached model for: {full_path}")
            return (self._CACHE[cache_key],)

        try:
            model_obj = torch.load(full_path, map_location=device)
        except Exception as e:
            raise RuntimeError(f"[ComfyUI-PTHLoader] Failed to load {full_path}: {e}")

        self._CACHE[cache_key] = model_obj
        print(f"[ComfyUI-PTHLoader] Loaded: {full_path}")
        return (model_obj,)
