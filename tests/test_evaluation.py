"""Unit tests for evaluation metrics, reports, and model evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from mambatransqr.evaluation import (  # noqa: E402
    BenchmarkRunner,
    Evaluator,
    MetricsManager,
    QRDecoderEvaluator,
    ReportGenerator,
    auc,
    compute_confusion_matrix,
    roc_curve,
    summarize,
)


def test_image_metrics_match_identical_images() -> None:
    """Image metrics report zero error and near-perfect similarity."""
    image = torch.ones(2, 3, 8, 8)
    metrics = MetricsManager(("mse", "mae", "ssim", "psnr")).compute(image, image)
    assert metrics["mse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["ssim"] == pytest.approx(1.0)
    assert metrics["psnr"] > 60


def test_classification_metrics_include_top_k() -> None:
    """Classification metrics calculate top-1 and top-5 accuracy."""
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0, 0.0]])
    targets = torch.tensor([4, 0])
    values = MetricsManager(("accuracy", "top1", "top5")).compute(logits, targets)
    assert values == {"accuracy": 1.0, "top1": 1.0, "top5": 1.0}


def test_evaluator_averages_batch_metrics() -> None:
    """Evaluator calculates mean metrics over mapping-style batches."""
    model = torch.nn.Identity()
    evaluator = Evaluator(model, MetricsManager(("mse",)), torch.device("cpu"))
    batches = [{"image": torch.ones(1, 1), "target": torch.zeros(1, 1)}]
    assert evaluator.evaluate(batches) == {"mse": 1.0}


def test_confusion_matrix_and_roc_are_computed() -> None:
    """Confusion matrix and ROC helpers return expected tensor shapes."""
    matrix = compute_confusion_matrix(torch.tensor([0, 1, 1]), torch.tensor([0, 1, 0]), 2)
    false_positive_rate, true_positive_rate, _ = roc_curve(
        torch.tensor([0.1, 0.9]), torch.tensor([0, 1])
    )
    assert matrix.tolist() == [[1, 1], [0, 1]]
    assert auc(false_positive_rate, true_positive_rate) >= 0


def test_reports_are_written(tmp_path: Path) -> None:
    """Report generator writes JSON, CSV, and Markdown report artifacts."""
    metrics = {"mse": 0.1, "psnr": 20.0}
    assert ReportGenerator.evaluation_json(metrics, tmp_path / "evaluation.json").is_file()
    assert ReportGenerator.benchmark_csv(metrics, tmp_path / "benchmark.csv").is_file()
    assert ReportGenerator.summary_markdown(metrics, metrics, tmp_path / "summary.md").is_file()


def test_statistics_summary_has_expected_mean() -> None:
    """Statistics helper calculates the expected sample mean."""
    assert summarize([1.0, 2.0, 3.0]).mean == pytest.approx(2.0)


def test_decoder_rejects_unknown_backend() -> None:
    """QR decoder validates requested backend names."""
    with pytest.raises(ValueError, match="backend"):
        QRDecoderEvaluator("unknown")


def test_benchmark_reports_latency_and_throughput() -> None:
    """Benchmark runner emits positive inference timing and throughput metrics."""
    model = torch.nn.Linear(2, 1)
    result = BenchmarkRunner(model, torch.device("cpu")).run(
        torch.ones(2, 2), warmup=0, iterations=1
    )
    assert result.latency.samples == 1
    assert result.throughput > 0
