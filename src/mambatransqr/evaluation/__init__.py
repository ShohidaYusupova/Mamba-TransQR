"""Evaluation, benchmark, decoding, reporting, and visualization utilities."""

from mambatransqr.evaluation.benchmark import BenchmarkResult, BenchmarkRunner
from mambatransqr.evaluation.confusion_matrix import compute_confusion_matrix
from mambatransqr.evaluation.decoder import DecodeResult, QRDecoderEvaluator
from mambatransqr.evaluation.evaluator import Evaluator
from mambatransqr.evaluation.latency import LatencyResult, measure_latency
from mambatransqr.evaluation.metrics import MetricsManager
from mambatransqr.evaluation.report import ReportGenerator
from mambatransqr.evaluation.roc import auc, roc_curve
from mambatransqr.evaluation.statistics import StatisticsResult, summarize
from mambatransqr.evaluation.visualizer import EvaluationVisualizer

__all__ = [
    "BenchmarkResult",
    "BenchmarkRunner",
    "DecodeResult",
    "EvaluationVisualizer",
    "Evaluator",
    "LatencyResult",
    "MetricsManager",
    "QRDecoderEvaluator",
    "ReportGenerator",
    "StatisticsResult",
    "auc",
    "compute_confusion_matrix",
    "measure_latency",
    "roc_curve",
    "summarize",
]
