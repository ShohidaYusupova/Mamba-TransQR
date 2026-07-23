"""Classification accuracy metrics."""
from torch import Tensor
def accuracy(logits: Tensor, targets: Tensor) -> Tensor:
    """Return top-1 classification accuracy."""
    return (logits.argmax(1) == targets).float().mean()
def top_k_accuracy(logits: Tensor, targets: Tensor, k: int) -> Tensor:
    """Return top-k classification accuracy."""
    return logits.topk(k, 1).indices.eq(targets.unsqueeze(1)).any(1).float().mean()
