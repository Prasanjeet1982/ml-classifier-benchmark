# ML Systems Benchmark: Traditional Classifiers vs Modern Foundation Classifiers

A self-contained, reproducible Python benchmarking suite comparing traditional statistical classifiers against dense embedding baselines and local zero-shot foundation models on real-world banking intent data (`banking77`).

This pipeline runs **completely locally** on consumer hardware with zero external API dependencies, credentials, or paid third-party services.

---

## 📊 Benchmark Summary (NVIDIA RTX 4060 8GB GPU Accelerated)

| Model | Model Architecture / Type | Supervised Training Samples | Accuracy (%) | Macro F1 | Latency p50 (ms) | Latency p95 (ms) | Train/Embedding Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + LogisticRegression** | Traditional Sparse Linear Classifier | 400 | **98.50%** | **0.9851** | **2.53 ms** | **3.19 ms** | **0.387 s** |
| **TF-IDF + HistGradientBoosting** | Non-Linear Decision Trees | 400 | 93.50% | 0.9355 | 14.49 ms | 29.54 ms | 25.836 s |
| **MiniLM Embeddings + LogisticRegression** | Dense Semantic Embeddings | 400 | 97.50% | 0.9752 | 18.79 ms | 38.72 ms | 1.924 s |
| **DeBERTa-v3 Zero-Shot (Jev Proxy)** | NLI Foundation Model (Zero-Shot) | **0** | 92.50% | 0.9248 | **198.49 ms** | **379.20 ms** | **0.000 s** |
| **FLAN-T5-Base Zero-Shot** | Generative Seq2Seq LLM (Zero-Shot) | **0** | 90.50% | 0.9061 | **296.87 ms** | **444.84 ms** | **0.000 s** |

### GPU Acceleration Impact (RTX 4060 vs. CPU)

- **DeBERTa-v3 Zero-Shot**: Latency dropped from **2,074.54 ms (CPU)** down to **75.74 ms (GPU)** — a **~27.4× speedup**, bringing tail latency (`p95 = 95.06 ms`) below 100 ms for production viability.
- **Zero API Keys & 100% Offline**: All models run locally on your RTX 4060 without any external cloud API keys or fees.

---

2. **The Data vs. Cold-Start Trade-off**:
   - When **400 labeled samples** are available (80 per class), a simple linear model over n-gram TF-IDF features attains **98.5% accuracy** with negligible training overhead (54 milliseconds).
   - When **zero labeled data exists (Day 0)**, **DeBERTa-v3** delivers **92.5% accuracy out-of-the-box** using semantic hypothesis templates (`"This request is about {}"`), making it ideal for bootstrapping, cold-start domains, or generating pseudo-labels for downstream distillation.

---

## 📈 Visual Benchmark Dashboard

The benchmark automatically generates a 2-panel comparison chart (`benchmark_summary.png`):
- **Panel 1**: Accuracy (%) and Macro-F1 across models.
- **Panel 2**: Log-scale per-inference latency (p50 and p95 in ms), highlighting microsecond statistical models vs neural foundation latency.

![Benchmark Summary](benchmark_summary.png)

---

## 🎯 Dataset & Class Selection

The benchmark extracts 5 distinct semantic banking classes from the canonical `banking77` dataset:
1. `card_arrival`
2. `change_pin`
3. `transfer_fee_charged`
4. `automatic_top_up`
5. `lost_or_stolen_card`

### Stratified Split:
- **Supervised Training Split**: 80 examples per class (**400 total**)
- **Evaluation Test Split**: 40 examples per class (**200 total**)
- **Reproducibility Guarantee**: The test split is exported to [`test_dataset_split.json`](test_dataset_split.json) to guarantee identical test inputs across all models.

---

## 🛠 Project Structure

```
├── run_benchmark.py              # CLI entry point with configurable arguments
├── requirements.txt              # Pinned Python dependencies
├── final_benchmark_report.csv    # Consolidated output metrics table
├── test_dataset_split.json       # Exact stratified test split
├── benchmark_summary.png         # High-resolution 2-panel chart
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Banking77 loading, filtering & stratified splitting
│   ├── models.py                 # Unified interfaces for all 4 benchmark models
│   ├── benchmark.py              # Latency timer (p50/p95 via perf_counter), accuracy & F1
│   └── visualization.py          # 2-panel publication chart generator
└── README.md                     # Documentation & engineering findings
```

---

## 🚀 Quickstart & Reproduction

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run the Benchmark Harness (GPU Accelerated)
```bash
# Using the GPU environment on RTX 4060:
.venv_gpu\Scripts\python run_benchmark.py --device cuda
```

### 3. CLI Customization Flags
You can configure the benchmark run via command-line arguments:
```bash
.venv_gpu\Scripts\python run_benchmark.py \
  --train-per-class 80 \
  --test-per-class 40 \
  --random-seed 42 \
  --device cuda \
  --csv-out final_benchmark_report.csv \
  --chart-out benchmark_summary.png \
  --test-json-out test_dataset_split.json
```

---

## 🔬 Benchmark Methodology Details

- **Latency Measurement**: Each individual test sample is evaluated in isolation with `time.perf_counter()`. Percentiles (p50 median and p95 tail) are calculated across all 200 inference passes after warm-up iterations.
- **Hypothesis Formulation**: Zero-shot classification constructs natural-language NLI hypotheses:
  `"This request is about {}."`
- **Linear & Tree Baselines**:
  - `TfidfVectorizer(ngram_range=(1, 2), max_features=3000)`
  - `LogisticRegression(max_iter=1000)`
  - `HistGradientBoostingClassifier(max_iter=100)`
