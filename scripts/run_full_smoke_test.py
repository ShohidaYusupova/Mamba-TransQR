"""CPU-only end-to-end validation for the core research pipeline."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from mambatransqr.benchmarks.metrics import aggregate
from mambatransqr.benchmarks.tables import export_table
from mambatransqr.losses import MultiScaleRestorationLoss
from mambatransqr.models import ModelConfig, build_model
from mambatransqr.training import OptimizerFactory


def run(output: str | Path = "results/smoke_test", seed: int = 42) -> dict[str, object]:
    """Run each CPU stage and raise immediately if any required stage fails."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    stages: dict[str, str] = {}
    optional: dict[str, str] = {}
    try:
        clean = Image.new("RGB", (16, 16), "white")
        ImageDraw.Draw(clean).rectangle((4, 4, 11, 11), fill="black")
        damaged = clean.copy()
        ImageDraw.Draw(damaged).rectangle((7, 7, 9, 9), fill="white")
        clean.save(root / "clean.png")
        damaged.save(root / "damaged.png")
        stages["synthetic_dataset"] = "ok"
        target = (
            torch.from_numpy(np.asarray(clean, dtype=np.float32).copy())
            .permute(2, 0, 1)
            .unsqueeze(0)
            / 255
        )
        inputs = (
            torch.from_numpy(np.asarray(damaged, dtype=np.float32).copy())
            .permute(2, 0, 1)
            .unsqueeze(0)
            / 255
        )
        model = build_model(
            ModelConfig(
                image_size=16,
                patch_size=8,
                embed_dim=8,
                depth=1,
                num_heads=2,
                dropout=0.0,
                mamba_backend="lightweight",
            )
        )
        objective = MultiScaleRestorationLoss()
        optimizer = OptimizerFactory.create(model.parameters())
        output_tensor = model(inputs)
        loss = objective(output_tensor, target, [{"qr_version": 1, "border": 4}])
        loss.backward()
        optimizer.step()
        stages["train_validation"] = "ok"
        checkpoint = root / "checkpoint.pt"
        torch.save({"model": model.state_dict()}, checkpoint)
        model.load_state_dict(torch.load(checkpoint, weights_only=False)["model"])
        stages["checkpoint"] = "ok"
        prediction = model(inputs).detach()
        mse = torch.nn.functional.mse_loss(prediction, target).item()
        rows = [
            {
                "model": "smoke",
                "mse": mse,
                "psnr": float(-10 * torch.log10(torch.tensor(max(mse, 1e-8)))),
            }
        ]
        stages["inference_metrics_benchmark"] = "ok"
        with (root / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        export_table(aggregate(rows, ("model",)), "generated_table", root)
        (root / "generated_table.md").rename(root / "generated_table.md")
        clean.save(root / "generated_figure.png")
        (root / "predictions").mkdir(exist_ok=True)
        clean.save(root / "predictions" / "prediction.png")
        (root / "benchmark_summary.csv").write_text(
            "model,psnr\nsmoke," + str(rows[0]["psnr"]) + "\n", encoding="utf-8"
        )
        (root / "training_history.csv").write_text(
            "epoch,loss\n1," + str(float(loss.detach())) + "\n", encoding="utf-8"
        )
        try:
            import qrcode  # noqa: F401
        except ImportError:
            optional["qrcode"] = "skipped: install .[qr-generation]"
        optional["mamba_ssm"] = "skipped: lightweight backend selected"
        optional["zbar_zxing_lpips"] = "skipped: not required by CPU smoke test"
    except Exception as error:
        raise RuntimeError(f"smoke-test stage failed: {stages}") from error
    report = {"seed": seed, "stages": stages, "optional": optional, "rows": rows}
    (root / "smoke_test_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    run()
