"""
Publication-quality visualization generator for ML benchmarking.
Produces a 2-panel chart comparing Accuracy/Macro-F1 and Log-scale Latency.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_benchmark_chart(
    report_df: pd.DataFrame,
    output_image_path: str = "benchmark_summary.png",
    dpi: int = 300,
) -> None:
    """
    Generates a 2-panel comparison chart:
      Panel 1: Accuracy (%) and Macro-F1 across models.
      Panel 2: Log-scale p50 and p95 Latency (ms) comparing statistical vs neural models.
    """
    # Set overall aesthetic style
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelweight": "semibold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.titlesize": 15,
            "figure.titleweight": "bold",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor="#fdfdfd")
    fig.patch.set_facecolor("#fdfdfd")

    # Shorten model names for clean plotting if needed
    name_map = {
        "TF-IDF + LogisticRegression": "TF-IDF + LogReg\n(400 samples)",
        "TF-IDF + HistGradientBoosting": "TF-IDF + HistGB\n(400 samples)",
        "MiniLM Embeddings + LogisticRegression": "MiniLM + LogReg\n(400 samples)",
        "DeBERTa-v3 Zero-Shot (Jev Proxy)": "DeBERTa Zero-Shot (Jev)\n(0 samples)",
        "FLAN-T5-Base Zero-Shot": "FLAN-T5 Zero-Shot\n(0 samples)",
        "NVIDIA/Stanford CLM-8B (4-bit)": "CLM-8B (4-bit)\n(0 samples)",
    }
    model_labels = [name_map.get(m, m) for m in report_df["Model"]]
    x = np.arange(len(model_labels))
    width = 0.35

    # -------------------------------------------------------------
    # Panel 1: Accuracy (%) & Macro-F1
    # -------------------------------------------------------------
    ax1 = axes[0]
    ax1.set_facecolor("#ffffff")
    ax1.grid(axis="y", linestyle="--", alpha=0.35, zorder=0, color="#888888")

    acc_vals = report_df["Accuracy (%)"].values
    f1_vals = [f * 100.0 for f in report_df["Macro F1"].values]  # scale to 100 for visual symmetry

    color_acc = "#2563EB"   # Rich Royal Blue
    color_f1 = "#0D9488"    # Modern Teal

    bars1 = ax1.bar(
        x - width / 2,
        acc_vals,
        width,
        label="Accuracy (%)",
        color=color_acc,
        alpha=0.92,
        edgecolor="#1D4ED8",
        linewidth=1.2,
        zorder=3,
    )
    bars2 = ax1.bar(
        x + width / 2,
        f1_vals,
        width,
        label="Macro F1 (×100)",
        color=color_f1,
        alpha=0.92,
        edgecolor="#0F766E",
        linewidth=1.2,
        zorder=3,
    )

    ax1.set_title("Classification Quality: Accuracy & Macro-F1", pad=14, color="#1e293b")
    ax1.set_ylabel("Score (%)", color="#1e293b")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, color="#334155")
    ax1.set_ylim(0, 125)
    ax1.legend(loc="upper right", framealpha=0.95, edgecolor="#cbd5e1")

    # Add numeric badges over bars
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#1e3a8a",
        )
    for i, bar in enumerate(bars2):
        raw_f1 = report_df["Macro F1"].values[i]
        h = bar.get_height()
        ax1.annotate(
            f"{raw_f1:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#134e4a",
        )

    # -------------------------------------------------------------
    # Panel 2: Log-Scale Latency (p50 and p95 in ms)
    # -------------------------------------------------------------
    ax2 = axes[1]
    ax2.set_facecolor("#ffffff")
    ax2.grid(axis="y", which="both", linestyle="--", alpha=0.35, zorder=0, color="#888888")

    p50_vals = report_df["Latency p50 (ms)"].values
    p95_vals = report_df["Latency p95 (ms)"].values

    color_p50 = "#F59E0B"   # Warm Amber
    color_p95 = "#EF4444"   # Crimson / Coral Red

    bars_p50 = ax2.bar(
        x - width / 2,
        p50_vals,
        width,
        label="Latency p50 (ms)",
        color=color_p50,
        alpha=0.90,
        edgecolor="#D97706",
        linewidth=1.2,
        zorder=3,
    )
    bars_p95 = ax2.bar(
        x + width / 2,
        p95_vals,
        width,
        label="Latency p95 (ms)",
        color=color_p95,
        alpha=0.90,
        edgecolor="#DC2626",
        linewidth=1.2,
        zorder=3,
    )

    ax2.set_yscale("log")
    ax2.set_title("Per-Inference Latency Profile (Log Scale, ms)", pad=14, color="#1e293b")
    ax2.set_ylabel("Inference Latency in ms (Log Scale)", color="#1e293b")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, color="#334155")
    ax2.legend(loc="upper left", framealpha=0.95, edgecolor="#cbd5e1")

    # Annotate latency values above bars
    for bar in bars_p50:
        h = bar.get_height()
        display_text = f"{h:.2f}ms" if h >= 1.0 else f"{h:.3f}ms"
        ax2.annotate(
            display_text,
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.0,
            fontweight="bold",
            color="#92400e",
        )
    for bar in bars_p95:
        h = bar.get_height()
        display_text = f"{h:.2f}ms" if h >= 1.0 else f"{h:.3f}ms"
        ax2.annotate(
            display_text,
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.0,
            fontweight="bold",
            color="#991b1b",
        )

    # Adjust vertical headroom on log plot
    curr_ymin, curr_ymax = ax2.get_ylim()
    ax2.set_ylim(curr_ymin * 0.5, curr_ymax * 8)

    # Global title & subtitle
    fig.suptitle(
        "Benchmarking Traditional vs Dense vs Zero-Shot Foundation Classifiers (banking77)",
        fontsize=16,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )

    # Spacing and layout polish
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    out_path = Path(output_image_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
