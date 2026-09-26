"""
Data loader and preprocessor for the banking77 benchmark dataset.
Extracts and stratifies 5 distinct semantic banking classes.
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
import numpy as np

logger = logging.getLogger(__name__)

TARGET_CLASSES: List[str] = [
    "card_arrival",
    "change_pin",
    "transfer_fee_charged",
    "automatic_top_up",
    "lost_or_stolen_card",
]


@dataclass
class BenchmarkData:
    """Holds train and test data splits for benchmarking."""
    train_texts: List[str]
    train_labels: List[str]
    test_texts: List[str]
    test_labels: List[str]
    classes: List[str]

    @property
    def num_train(self) -> int:
        return len(self.train_texts)

    @property
    def num_test(self) -> int:
        return len(self.test_texts)


def load_raw_banking77_dataset():
    """
    Attempts to load banking77 from Hugging Face Hub.
    Uses 'banking77' if available, falling back to 'mteb/banking77'
    for modern datasets library compatibility.
    """
    for dataset_id in ["banking77", "mteb/banking77", "PolyAI/banking77"]:
        try:
            logger.info(f"Attempting to load dataset from: '{dataset_id}'...")
            ds = load_dataset(dataset_id)
            logger.info(f"Successfully loaded '{dataset_id}'.")
            return ds
        except Exception as e:
            logger.warning(f"Failed to load '{dataset_id}': {e}")
    raise RuntimeError("Could not load banking77 from any repository candidate.")


def prepare_benchmark_dataset(
    target_classes: Optional[List[str]] = None,
    train_per_class: int = 80,
    test_per_class: int = 40,
    random_seed: int = 42,
    output_test_json_path: Optional[str] = "test_dataset_split.json",
) -> BenchmarkData:
    """
    Filters banking77 down to the specified classes and stratifies into:
      - Train: train_per_class examples per class (e.g., 80 * 5 = 400)
      - Test: test_per_class examples per class (e.g., 40 * 5 = 200)

    Exports the test split to `test_dataset_split.json` for reproducibility.
    """
    if target_classes is None:
        target_classes = TARGET_CLASSES

    raw_ds = load_raw_banking77_dataset()

    # Determine label mapping
    # Check if 'label_text' is directly available or if 'label' needs class label lookup
    def extract_rows(split_name: str) -> List[Tuple[str, str]]:
        split = raw_ds[split_name]
        has_label_text = "label_text" in split.features or "label_text" in split.column_names

        if has_label_text:
            texts = split["text"]
            labels = split["label_text"]
        else:
            feature_names = split.features["label"].names
            texts = split["text"]
            labels = [feature_names[idx] for idx in split["label"]]

        rows = []
        for text, lbl in zip(texts, labels):
            if lbl in target_classes:
                rows.append((str(text).strip(), str(lbl)))
        return rows

    train_rows = extract_rows("train")
    test_rows = extract_rows("test")

    logger.info(f"Extracted {len(train_rows)} filtered train rows and {len(test_rows)} filtered test rows.")

    rng = np.random.RandomState(random_seed)

    # Stratified selection for train
    selected_train_texts: List[str] = []
    selected_train_labels: List[str] = []

    for cls in target_classes:
        cls_pool = [t for t, l in train_rows if l == cls]
        if len(cls_pool) < train_per_class:
            raise ValueError(
                f"Class '{cls}' only has {len(cls_pool)} train examples, but {train_per_class} requested."
            )
        indices = rng.choice(len(cls_pool), size=train_per_class, replace=False)
        for idx in indices:
            selected_train_texts.append(cls_pool[idx])
            selected_train_labels.append(cls)

    # Stratified selection for test
    selected_test_texts: List[str] = []
    selected_test_labels: List[str] = []

    for cls in target_classes:
        cls_pool = [t for t, l in test_rows if l == cls]
        if len(cls_pool) < test_per_class:
            raise ValueError(
                f"Class '{cls}' only has {len(cls_pool)} test examples, but {test_per_class} requested."
            )
        # Sort or deterministic choice
        indices = rng.choice(len(cls_pool), size=test_per_class, replace=False)
        for idx in indices:
            selected_test_texts.append(cls_pool[idx])
            selected_test_labels.append(cls)

    # Save exact test set to test_dataset_split.json
    if output_test_json_path:
        test_export = {
            "metadata": {
                "dataset": "banking77",
                "classes": target_classes,
                "samples_per_class": test_per_class,
                "total_samples": len(selected_test_texts),
                "random_seed": random_seed,
            },
            "samples": [
                {
                    "id": i,
                    "text": text,
                    "label": label,
                }
                for i, (text, label) in enumerate(zip(selected_test_texts, selected_test_labels))
            ],
        }
        out_path = Path(output_test_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(test_export, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(selected_test_texts)} test examples to '{out_path.resolve()}'.")

    return BenchmarkData(
        train_texts=selected_train_texts,
        train_labels=selected_train_labels,
        test_texts=selected_test_texts,
        test_labels=selected_test_labels,
        classes=target_classes,
    )
