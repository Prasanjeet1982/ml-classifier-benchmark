"""
Benchmark execution engine for measuring classification accuracy, Macro-F1,
and per-inference latency (p50 and p95) using high-precision counters.
"""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
import torch
from tqdm import tqdm

from src.data_loader import BenchmarkData
from src.models import BaseBenchmarkClassifier

logger = logging.getLogger(__name__)


@dataclass
class ModelBenchmarkResult:
    """Consolidated performance and latency metrics for a model."""
    model_name: str
    model_type: str
    supervised_samples: int
    train_time_sec: float
    accuracy_pct: float
    macro_f1: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_mean_ms: float
    latency_min_ms: float
    latency_max_ms: float
    num_test_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def benchmark_single_model(
    model: BaseBenchmarkClassifier,
    data: BenchmarkData,
    warmup_runs: int = 3,
) -> ModelBenchmarkResult:
    """
    Trains (if supervised) and benchmarks a single model across the test set.
    Measures per-sample latency in milliseconds using time.perf_counter().
    Synchronizes CUDA queues for precision on GPU.
    """
    logger.info(f"\n=======================================================")
    logger.info(f"Benchmarking: {model.name}")
    logger.info(f"Model Type: {model.model_type}")
    logger.info(f"Supervised Samples: {model.supervised_samples_needed}")
    logger.info(f"=======================================================")

    is_cuda = torch.cuda.is_available() and getattr(model, "device", "cpu") != "cpu"

    # 1. Training / Fitting Phase
    if model.supervised_samples_needed > 0:
        logger.info(f"Fitting model on {len(data.train_texts)} training examples...")
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        train_time = model.fit(data.train_texts, data.train_labels)
        if is_cuda:
            torch.cuda.synchronize()
        logger.info(f"Training / embedding completed in {train_time:.4f} seconds.")
    else:
        logger.info("Zero-shot model: Skipping training phase (0 supervised samples).")
        train_time = model.fit([], [])

    # 2. Warm-up Phase
    logger.info(f"Executing {warmup_runs} warm-up inference calls...")
    for i in range(min(warmup_runs, len(data.test_texts))):
        _ = model.predict_single(data.test_texts[i])
    if is_cuda:
        torch.cuda.synchronize()

    # 3. Per-Inference Latency & Prediction Evaluation
    logger.info(f"Evaluating {len(data.test_texts)} test queries with individual latency measurement...")
    predictions: List[str] = []
    latencies_ms: List[float] = []

    for text in tqdm(data.test_texts, desc=f"Eval {model.name}", unit="query"):
        if is_cuda:
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        pred = model.predict_single(text)
        if is_cuda:
            torch.cuda.synchronize()
        t_elapsed = (time.perf_counter() - t_start) * 1000.0  # convert to ms

        latencies_ms.append(t_elapsed)
        predictions.append(pred)

    # 4. Metrics Computation
    acc = accuracy_score(data.test_labels, predictions) * 100.0
    macro_f1 = f1_score(data.test_labels, predictions, average="macro")

    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    mean_lat = float(np.mean(latencies_ms))
    min_lat = float(np.min(latencies_ms))
    max_lat = float(np.max(latencies_ms))

    result = ModelBenchmarkResult(
        model_name=model.name,
        model_type=model.model_type,
        supervised_samples=model.supervised_samples_needed,
        train_time_sec=round(train_time, 4),
        accuracy_pct=round(acc, 2),
        macro_f1=round(macro_f1, 4),
        latency_p50_ms=round(p50, 4),
        latency_p95_ms=round(p95, 4),
        latency_mean_ms=round(mean_lat, 4),
        latency_min_ms=round(min_lat, 4),
        latency_max_ms=round(max_lat, 4),
        num_test_samples=len(data.test_texts),
    )

    logger.info(
        f"Results for {model.name}:\n"
        f"  Accuracy:      {result.accuracy_pct:.2f}%\n"
        f"  Macro-F1:      {result.macro_f1:.4f}\n"
        f"  Latency p50:   {result.latency_p50_ms:.3f} ms\n"
        f"  Latency p95:   {result.latency_p95_ms:.3f} ms\n"
        f"  Train Time:    {result.train_time_sec:.4f} s\n"
    )

    return result


def run_full_benchmark(
    models: List[BaseBenchmarkClassifier],
    data: BenchmarkData,
    output_csv_path: Optional[str] = "final_benchmark_report.csv",
) -> pd.DataFrame:
    """
    Runs benchmarks across all specified models, compiles results into a DataFrame,
    and exports to CSV.
    """
    results: List[ModelBenchmarkResult] = []

    for model in models:
        res = benchmark_single_model(model, data)
        results.append(res)

    df = pd.DataFrame([r.to_dict() for r in results])

    # Reorder / format columns nicely for output
    column_rename = {
        "model_name": "Model",
        "model_type": "Model Type",
        "supervised_samples": "Supervised Samples Needed",
        "accuracy_pct": "Accuracy (%)",
        "macro_f1": "Macro F1",
        "latency_p50_ms": "Latency p50 (ms)",
        "latency_p95_ms": "Latency p95 (ms)",
        "latency_mean_ms": "Latency Mean (ms)",
        "train_time_sec": "Train/Embedding Time (s)",
        "num_test_samples": "Test Samples Evaluated",
    }
    present_cols = [c for c in column_rename.keys() if c in df.columns]
    report_df = df[present_cols].rename(columns=column_rename)

    if output_csv_path:
        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(out_path, index=False)
        logger.info(f"Saved benchmark report to '{out_path.resolve()}'.")

    return report_df
