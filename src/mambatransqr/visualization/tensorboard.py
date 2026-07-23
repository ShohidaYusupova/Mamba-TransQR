"""TensorBoard visualization logger."""

from pathlib import Path


class TensorBoardVisualizer:
    """Small TensorBoard scalar/image wrapper."""

    def __init__(self, log_dir: str | Path) -> None:
        """Create optional TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise ImportError("TensorBoard requires tensorboard.") from error
        self.writer = SummaryWriter(str(log_dir))

    def scalar(self, name: str, value: float, step: int) -> None:
        """Write a scalar."""
        self.writer.add_scalar(name, value, step)

    def close(self) -> None:
        """Close writer."""
        self.writer.close()
