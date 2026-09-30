"""
CLI Entry Point: Benchmarking Harness for Traditional vs Foundation Classifiers.

Executes stratified dataset loading, baseline training, inference benchmarking,
latency profiling (p50/p95), and artifact generation (CSV report + 2-panel chart).
"""

import argparse
import logging
from pathlib import Path
import sys

from src.benchmark import run_full_benchmark
from src.data_loader import TARGET_CLASSES, prepare_benchmark_dataset
from src.models import (
    Clm8bZeroShotModel,
    DebertaZeroShotModel,
    DenseEmbeddingLogisticRegressionModel,
    FlanT5ZeroShotModel,
    TfidfHistGradientBoostingModel,
    TfidfLogisticRegressionModel,
)
from src.visualization import generate_benchmark_chart

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_benchmark")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Traditional Classifiers vs Modern Foundation Models on banking77."
    )
    parser.add_argument(
        "--train-per-class",
        type=int,
        default=80,
        help="Supervised training samples per class (default: 80, 400 total across 5 classes).",
    )
    parser.add_argument(
        "--test-per-class",
        type=int,
        default=40,
        help="Test evaluation samples per class (default: 40, 200 total across 5 classes).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for deterministic data splitting (default: 42).",
    )
    parser.add_argument(
        "--test-json-out",
        type=str,
        default="test_dataset_split.json",
        help="Path to save the test split json (default: test_dataset_split.json).",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default="final_benchmark_report.csv",
        help="Path to output consolidated CSV report (default: final_benchmark_report.csv).",
    )
    parser.add_argument(
        "--chart-out",
        type=str,
        default="benchmark_summary.png",
        help="Path to output 2-panel comparison chart (default: benchmark_summary.png).",
    )
    import torch
    default_dev = "cuda" if torch.cuda.is_available() else "cpu"

    parser.add_argument(
        "--device",
        type=str,
        default=default_dev,
        help=f"Execution device for neural models ('cpu' or 'cuda', default: '{default_dev}').",
    )
    parser.add_argument(
        "--include-clm",
        action="store_true",
        default=False,
        help="Include NVIDIA/Stanford CLM-8B (4-bit NF4 quantized) in the benchmark.",
    )
    return parser.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()

    import torch
    gpu_info = (
        f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB VRAM)"
        if torch.cuda.is_available() and args.device.startswith("cuda")
        else "CPU (No accelerator)"
    )

    print("\n" + "=" * 80)
    print(" [BENCHMARK] ML SYSTEMS BENCHMARK: TRADITIONAL VS FOUNDATION CLASSIFIERS")
    print("=" * 80)
    print(f"Target Semantic Classes ({len(TARGET_CLASSES)}): {', '.join(TARGET_CLASSES)}")
    print(f"Train samples per class: {args.train_per_class} (Total: {args.train_per_class * len(TARGET_CLASSES)})")
    print(f"Test samples per class:  {args.test_per_class} (Total: {args.test_per_class * len(TARGET_CLASSES)})")
    print(f"Random seed:             {args.random_seed}")
    print(f"Execution Accelerator:   {gpu_info}")
    print("=" * 80 + "\n")

    # Step 1: Load and stratify dataset
    logger.info("Loading banking77 dataset and creating stratified splits...")
    data = prepare_benchmark_dataset(
        target_classes=TARGET_CLASSES,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
        random_seed=args.random_seed,
        output_test_json_path=args.test_json_out,
    )
    logger.info(
        f"Data prepared: {data.num_train} train rows, {data.num_test} test rows across {len(data.classes)} classes."
    )

    # Step 2: Initialize models
    logger.info("Initializing baseline and zero-shot models...")
    models = [
        TfidfLogisticRegressionModel(random_state=args.random_seed),
        TfidfHistGradientBoostingModel(random_state=args.random_seed),
        DenseEmbeddingLogisticRegressionModel(
            model_name="all-MiniLM-L6-v2",
            random_state=args.random_seed,
            device=args.device,
        ),
        DebertaZeroShotModel(
            model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
            target_classes=TARGET_CLASSES,
            hypothesis_template="This request is about {}.",
            device=args.device,
        ),
        FlanT5ZeroShotModel(
            model_name="google/flan-t5-base",
            target_classes=TARGET_CLASSES,
            device=args.device,
        ),
    ]

    if args.include_clm:
        logger.info("Including NVIDIA/Stanford CLM-8B (4-bit NF4) in the benchmark suite...")
        models.append(
            Clm8bZeroShotModel(
                base_model_name="Qwen/Qwen3-8B",
                target_classes=TARGET_CLASSES,
                device=args.device,
            )
        )

    # Step 3: Run full benchmark loop
    logger.info("Starting execution of benchmark pipeline...")
    report_df = run_full_benchmark(models=models, data=data, output_csv_path=args.csv_out)

    # Step 4: Generate visual summary
    logger.info("Generating publication-quality 2-panel chart...")
    generate_benchmark_chart(report_df, output_image_path=args.chart_out)
    logger.info(f"Summary visualization saved to: {Path(args.chart_out).resolve()}")

    # Step 5: Display final table summary in console
    print("\n" + "=" * 96)
    print(" [REPORT] CONSOLIDATED BENCHMARK REPORT")
    print("=" * 96)
    print(report_df.to_string(index=False))
    print("=" * 96)
    print(f"[OK] Exact test split saved to:      {Path(args.test_json_out).resolve()}")
    print(f"[OK] Final benchmark CSV saved to:   {Path(args.csv_out).resolve()}")
    print(f"[OK] Benchmark chart saved to:       {Path(args.chart_out).resolve()}")
    print("=" * 96 + "\n")


if __name__ == "__main__":
    main()
