# Evaluation

Use `Evaluator` and `MetricsManager` to compute image or classification metrics
over a loader. `BenchmarkRunner` reports latency, throughput, and CPU/GPU memory.
`ReportGenerator` writes JSON, CSV, and Markdown artifacts.

QR decode evaluation is available through `QRDecoderEvaluator`; install
`pyzbar` (ZBar) or `zxing-cpp` to enable its optional backends. Without one of
these backends, payload decode rate is unavailable. The Phase 3 experiment's
threshold-based decode proxy is a different, explicitly labeled metric and
must not be presented as payload recovery.

Benchmark recipes write measured raw values plus reproducibility metadata,
grouped summaries, publication tables, and optional PDF/PNG figures. Baselines
must declare a compatible local checkpoint and provenance. Unavailable models
are marked `N/A`, never substituted or assigned estimated values.

Latency is reported in milliseconds with its batch size, warmup count, timed
iteration count, device type, and available runtime identity. CPU results are
not portable hardware constants; comparisons require the same measurement
process and hardware context.
