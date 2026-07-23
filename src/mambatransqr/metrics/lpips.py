"""Optional LPIPS metric."""

import torch
from torch import Tensor


def lpips_score(predictions: Tensor, targets: Tensor) -> Tensor:
    """Return LPIPS perceptual distance."""
    try:
        import lpips
    except ImportError as error:
        raise ImportError("LPIPS requires the optional lpips package.") from error
    model = lpips.LPIPS(net="alex").to(predictions.device).eval()
    with torch.no_grad():
        return model(predictions * 2 - 1, targets * 2 - 1).mean()
