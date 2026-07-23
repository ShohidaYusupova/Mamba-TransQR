# Evaluation

Use `Evaluator` and `MetricsManager` to compute image or classification metrics
over a loader. `BenchmarkRunner` reports latency, throughput, and CPU/GPU memory.
`ReportGenerator` writes JSON, CSV, and Markdown artifacts.

QR decode evaluation is available through `QRDecoderEvaluator`; install either
`pyzbar` (ZBar) or `zxing-cpp` to enable its optional backends.
