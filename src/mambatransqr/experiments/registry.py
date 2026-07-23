"""Experiment registry."""

from pathlib import Path


class ExperimentRegistry:
    """Discover experiments stored under a common root."""

    def __init__(self, root: str | Path) -> None:
        """Store registry root."""
        self.root = Path(root)

    def list(self) -> list[Path]:
        """Return registered experiment directories."""
        if not self.root.exists():
            return []
        return sorted(path for path in self.root.iterdir() if path.is_dir())
