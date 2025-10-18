import os
import sys
import types
import importlib.util
import traceback

# ---- Comfy expects these at module scope ----
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ---- Paths ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_NODES_DIR = os.path.join(_THIS_DIR, "nodes")

# Ensure package dirs are importable
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _NODES_DIR not in sys.path:
    sys.path.insert(0, _NODES_DIR)

# ---- Create a package hierarchy so relative imports (.utils) work ----
PKG_ROOT = "comfyui_cn_pre"
PKG_NODES = f"{PKG_ROOT}.nodes"

def _ensure_pkg(name: str, path: str, file_hint: str):
    if name in sys.modules:
        return sys.modules[name]
    m = types.ModuleType(name)
    m.__path__ = [path]
    m.__file__ = file_hint
    m.__package__ = name
    try:
        import importlib.machinery as _mach
        m.__spec__ = _mach.ModuleSpec(name=name, loader=None, is_package=True)
        m.__spec__.submodule_search_locations = [path]
    except Exception:
        pass
    sys.modules[name] = m
    return m

_ = _ensure_pkg(PKG_ROOT, _THIS_DIR, os.path.join(_THIS_DIR, "__init__.py"))
_ = _ensure_pkg(PKG_NODES, _NODES_DIR, os.path.join(_NODES_DIR, "__init__.py"))

def _load_as(module_name: str, file_path: str):
    """Load a file and register it in sys.modules under module_name."""
    try:
        if not os.path.isfile(file_path):
            print(f"[ComfyUI-CN-Pre] Missing file: {file_path}")
            return None
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            print(f"[ComfyUI-CN-Pre] Could not create spec for {file_path}")
            return None
        module = importlib.util.module_from_spec(spec)
        parent = module_name.rpartition(".")[0]
        module.__package__ = parent if parent else module_name
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception:
        print(f"[ComfyUI-CN-Pre] Skipped loading {file_path} as {module_name}:\n{traceback.format_exc()}")
        sys.modules.pop(module_name, None)
        return None

def _register_node(file_basename: str, class_name: str, key: str, display_name: str):
    """
    Load nodes/<file_basename>, fetch <class_name>, and register into
    NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS with <key>.
    Prioritizes module's NODE_CLASS_MAPPINGS.
    """
    file_path = os.path.join(_NODES_DIR, file_basename)
    stem = os.path.splitext(file_basename)[0]
    mod_name = f"{PKG_NODES}.{stem}"

    module = _load_as(mod_name, file_path)
    if module is None:
        print(f"[ComfyUI-CN-Pre] Failed to load module {file_basename}")
        return

    try:
        mod_map = getattr(module, "NODE_CLASS_MAPPINGS", None)
        mod_names = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", None)
        if isinstance(mod_map, dict) and len(mod_map) > 0:
            for k_mod, cls_mod in mod_map.items():
                NODE_CLASS_MAPPINGS[k_mod] = cls_mod
                if isinstance(mod_names, dict) and k_mod in mod_names:
                    NODE_DISPLAY_NAME_MAPPINGS[k_mod] = mod_names[k_mod]
            print(f"[ComfyUI-CN-Pre] Merged {len(mod_map)} class(es) from {file_basename}")
            return
    except Exception:
        print(f"[ComfyUI-CN-Pre] Failed to merge mappings from {file_basename}:\n{traceback.format_exc()}")

    try:
        cls = getattr(module, class_name)
        NODE_CLASS_MAPPINGS[key] = cls
        NODE_DISPLAY_NAME_MAPPINGS[key] = display_name
        print(f"[ComfyUI-CN-Pre] Loaded: {key} -> {file_basename}:{class_name}")
    except Exception:
        print(f"[ComfyUI-CN-Pre] '{class_name}' not found in {file_basename} and no valid NODE_CLASS_MAPPINGS")

# ---- Load utils first so `.utils` resolves when nodes import it at import-time ----
_utils_path = os.path.join(_NODES_DIR, "utils.py")
if os.path.isfile(_utils_path):
    _load_as(f"{PKG_NODES}.utils", _utils_path)
else:
    print(f"[ComfyUI-CN-Pre] Warning: utils.py not found at {_utils_path}")

# ---- Register existing nodes ----
_register_node("pre_canny.py", "PreCannyWithLoaders",
               "CN Pre+Loaders: Canny", "ControlNet Pre (+Model/ControlNet): Canny")
_register_node("pre_depth_midas.py", "PreDepthMiDaSWithLoaders",
               "CN Pre+Loaders: Depth (MiDaS)", "ControlNet Pre (+Model/ControlNet): Depth (MiDaS)")
_register_node("pre_depth_depthanything.py", "PreDepthAnythingWithLoaders",
               "CN Pre+Loaders: Depth (DepthAnything)", "ControlNet Pre (+Model/ControlNet): Depth (DepthAnything)")
_register_node("normalize.py", "ControlNormalize",
               "CN Pre: Normalize", "ControlNet Normalize")
_register_node("resize.py", "ResizeForControl",
               "CN Pre: Resize", "Resize for ControlNet")
_register_node("schedule.py", "ControlSchedule",
               "CN Schedule", "ControlNet Strength Schedule")
_register_node("pth_loader.py", "PTHLoader",
               "PTH Loader", "Generic .pth Loader")
_register_node("pre_lineart.py", "PreLineArt",
               "CN Pre+Loaders: LineArt", "ControlNet Pre (+Model/ControlNet): LineArt")
_register_node("pre_mlsd.py", "PreMLSDWithLoaders",
               "CN Pre+Loaders: MLSD", "ControlNet Pre (+Model/ControlNet): MLSD")
_register_node("pre_segmentation.py", "PreSegmentationWithLoaders",
               "CN Pre+Loaders: Segmentation", "ControlNet Pre (+Model/ControlNet): Segmentation")
_register_node("levels.py", "ControlLevels",
               "CN Pre: Levels", "Levels (Black/White/Gamma)")
_register_node(
    "ollama_vlm_captioner.py", "OllamaVLMCaptioner",
    "CN VLM: Ollama Captioner", "CN VLM: Ollama Captioner"
)
_register_node(
    "ollama_list_models.py", "OllamaListModels",
    "CN VLM: Ollama List Models", "CN VLM: Ollama List Models"
)