"""
NutriVision AI — PyTorch / Torchvision Compatibility Layer
Resolves pre-release PyTorch 2.14 / Torchvision 0.29 C++ NMS operator registration
anomalies on Windows x86_64 without altering package installations.
"""

import sys
import torch

def apply_torch_patch():
    """Defines missing TorchScript dispatcher bindings and installs pure-PyTorch NMS if required."""
    # First attempt to import torchvision natively so C++ operators register cleanly
    try:
        import torchvision
        import torchvision.ops.boxes
    except Exception:
        pass

    # Check if native torchvision NMS works without intervention
    native_nms_available = False
    try:
        if "torchvision" in sys.modules and hasattr(torchvision.ops, "nms"):
            _b = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
            _s = torch.tensor([1.0], dtype=torch.float32)
            _res = torchvision.ops.nms(_b, _s, 0.5)
            if _res is not None:
                native_nms_available = True
    except Exception:
        native_nms_available = False

    if not native_nms_available:
        try:
            torch.library.define("torchvision::nms", "(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
        except Exception:
            pass

        try:
            torch.library.define("torchvision::qnms", "(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
        except Exception:
            pass

        def pure_pytorch_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
            if boxes.numel() == 0:
                return torch.empty((0,), dtype=torch.int64, device=boxes.device)
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)
            order = scores.argsort(descending=True)
            keep = []
            while order.numel() > 0:
                if order.numel() == 1:
                    keep.append(order.item())
                    break
                i = order[0].item()
                keep.append(i)
                xx1 = torch.maximum(x1[i], x1[order[1:]])
                yy1 = torch.maximum(y1[i], y1[order[1:]])
                xx2 = torch.minimum(x2[i], x2[order[1:]])
                yy2 = torch.minimum(y2[i], y2[order[1:]])
                w = torch.clamp(xx2 - xx1, min=0.0)
                h = torch.clamp(yy2 - yy1, min=0.0)
                inter = w * h
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
                inds = torch.where(ovr <= iou_threshold)[0]
                order = order[inds + 1]
            return torch.tensor(keep, dtype=torch.int64, device=boxes.device)

        try:
            torchvision.ops.nms = pure_pytorch_nms
            torchvision.ops.boxes.nms = pure_pytorch_nms
        except Exception:
            pass

# Automatically execute on import
apply_torch_patch()
