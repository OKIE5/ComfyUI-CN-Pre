import torch, numpy as np
from PIL import Image

def to_bchw(bhwc): return bhwc.permute(0,3,1,2).contiguous()
def to_bhwc(bchw): return bchw.permute(0,2,3,1).contiguous()

def pil_list_from_bhwc(x: torch.Tensor):
    arr = (x.clamp(0,1).cpu().numpy()*255).astype(np.uint8)  # [B,H,W,C]
    return [Image.fromarray(arr[i]) for i in range(arr.shape[0])]

def bhwc_from_pil_list(pils):
    arr = np.stack([np.asarray(p).astype(np.float32)/255.0 for p in pils], 0)
    return torch.from_numpy(arr)  # [B,H,W,C]

def ensure_3ch_bhwc(img: torch.Tensor):
    assert img.ndim == 4
    return img.repeat(1,1,1,3) if img.shape[-1]==1 else img

def resize_bhwc(img: torch.Tensor, h: int, w: int):
    import torch.nn.functional as F
    return to_bhwc(F.interpolate(to_bchw(img), size=(h,w), mode="bilinear", align_corners=False))

def nearest_multiple_of_8(n: int) -> int:
    return max(8, (n // 8) * 8)
