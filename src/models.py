"""
Unified model abstractions for Traditional Classifiers, Dense Embedding Baselines,
and Local Zero-Shot Foundation Models.
"""

from abc import ABC, abstractmethod
import logging
import time
from typing import Dict, List, Optional

from huggingface_hub import hf_hub_download
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

logger = logging.getLogger(__name__)


class BaseBenchmarkClassifier(ABC):
    """Abstract base class for all benchmark classifiers."""

    name: str
    model_type: str
    supervised_samples_needed: int

    @abstractmethod
    def fit(self, texts: List[str], labels: List[str]) -> float:
        """
        Fits or initializes the model.
        Returns training / embedding elapsed time in seconds.
        """
        pass

    @abstractmethod
    def predict_single(self, text: str) -> str:
        """Predicts class label for a single input string."""
        pass

    def predict_batch(self, texts: List[str]) -> List[str]:
        """Default batch prediction calling predict_single."""
        return [self.predict_single(t) for t in texts]


class TfidfLogisticRegressionModel(BaseBenchmarkClassifier):
    """
    Traditional Baseline A:
    TF-IDF (ngram_range=(1,2), max_features=3000) + LogisticRegression(max_iter=1000).
    """

    def __init__(self, random_state: int = 42):
        self.name = "TF-IDF + LogisticRegression"
        self.model_type = "Traditional (Sparse Linear)"
        self.supervised_samples_needed = 400
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
        self.classifier = LogisticRegression(max_iter=1000, random_state=random_state)
        self._is_fitted = False

    def fit(self, texts: List[str], labels: List[str]) -> float:
        t0 = time.perf_counter()
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self._is_fitted = True
        return time.perf_counter() - t0

    def predict_single(self, text: str) -> str:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")
        X = self.vectorizer.transform([text])
        return str(self.classifier.predict(X)[0])

    def predict_batch(self, texts: List[str]) -> List[str]:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")
        X = self.vectorizer.transform(texts)
        return [str(l) for l in self.classifier.predict(X)]


class TfidfHistGradientBoostingModel(BaseBenchmarkClassifier):
    """
    Traditional Baseline B:
    TF-IDF (ngram_range=(1,2), max_features=3000) + HistGradientBoostingClassifier(max_iter=100).
    """

    def __init__(self, random_state: int = 42):
        self.name = "TF-IDF + HistGradientBoosting"
        self.model_type = "Traditional (Gradient Boosted Trees)"
        self.supervised_samples_needed = 400
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
        self.classifier = HistGradientBoostingClassifier(
            max_iter=100, random_state=random_state
        )
        self._is_fitted = False

    def fit(self, texts: List[str], labels: List[str]) -> float:
        t0 = time.perf_counter()
        X_sparse = self.vectorizer.fit_transform(texts)
        X_dense = X_sparse.toarray()
        self.classifier.fit(X_dense, labels)
        self._is_fitted = True
        return time.perf_counter() - t0

    def predict_single(self, text: str) -> str:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")
        X_sparse = self.vectorizer.transform([text])
        X_dense = X_sparse.toarray()
        return str(self.classifier.predict(X_dense)[0])

    def predict_batch(self, texts: List[str]) -> List[str]:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")
        X_sparse = self.vectorizer.transform(texts)
        X_dense = X_sparse.toarray()
        return [str(l) for l in self.classifier.predict(X_dense)]


class DenseEmbeddingLogisticRegressionModel(BaseBenchmarkClassifier):
    """
    Dense Embedding Baseline:
    SentenceTransformer('all-MiniLM-L6-v2') + LogisticRegression(max_iter=1000).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        random_state: int = 42,
        device: Optional[str] = None,
    ):
        self.name = "MiniLM Embeddings + LogisticRegression"
        self.model_type = "Dense Embedding (Few-Shot/Supervised)"
        self.supervised_samples_needed = 400
        self.model_name = model_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.embedding_model = SentenceTransformer(model_name, device=device)
        self.classifier = LogisticRegression(max_iter=1000, random_state=random_state)
        self._is_fitted = False

    def fit(self, texts: List[str], labels: List[str]) -> float:
        t0 = time.perf_counter()
        # Compute embeddings for 400 training samples
        embeddings = self.embedding_model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        )
        # Train Logistic Regression on dense embeddings
        self.classifier.fit(embeddings, labels)
        self._is_fitted = True
        return time.perf_counter() - t0

    def predict_single(self, text: str) -> str:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")
        # Single query embedding + linear classifier inference
        emb = self.embedding_model.encode(
            [text], show_progress_bar=False, convert_to_numpy=True
        )
        return str(self.classifier.predict(emb)[0])

    def predict_batch(self, texts: List[str]) -> List[str]:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")
        embeddings = self.embedding_model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        )
        return [str(l) for l in self.classifier.predict(embeddings)]


class DebertaZeroShotModel(BaseBenchmarkClassifier):
    """
    Local Zero-Shot Foundation Classifier:
    Hugging Face zero-shot-classification pipeline with MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli.
    Evaluates zero-shot with 0 training rows using hypothesis template:
    "This request is about {}"
    """

    def __init__(
        self,
        model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        target_classes: Optional[List[str]] = None,
        hypothesis_template: str = "This request is about {}.",
        device: Optional[str] = None,
    ):
        self.name = "DeBERTa-v3 Zero-Shot (Jev Proxy)"
        self.model_type = "Zero-Shot Foundation Model (Jev Proxy)"
        self.supervised_samples_needed = 0
        self.model_name = model_name
        self.hypothesis_template = hypothesis_template

        # Set up candidate labels and mapping between natural English and raw class labels
        if target_classes is None:
            target_classes = [
                "card_arrival",
                "change_pin",
                "transfer_fee_charged",
                "automatic_top_up",
                "lost_or_stolen_card",
            ]
        self.target_classes = target_classes

        # Natural phrase mapping: e.g. "card_arrival" -> "card arrival"
        self.clean_to_raw: Dict[str, str] = {
            c.replace("_", " "): c for c in target_classes
        }
        self.candidate_labels = list(self.clean_to_raw.keys())

        # Determine device and dtype (float16 on CUDA for Tensor Core acceleration)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if device == "cuda" or (isinstance(device, str) and device.startswith("cuda")):
            device_arg = 0
            model_kwargs = {"torch_dtype": torch.float16}
        else:
            device_arg = -1
            model_kwargs = {}

        self.pipeline = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device_arg,
            model_kwargs=model_kwargs,
        )

    def fit(self, texts: List[str], labels: List[str]) -> float:
        # Zero-shot model requires ZERO training data or supervised samples
        # Training / Embedding time is 0.0 seconds
        return 0.0

    def predict_single(self, text: str) -> str:
        res = self.pipeline(
            text,
            candidate_labels=self.candidate_labels,
            hypothesis_template=self.hypothesis_template,
            multi_label=False,
        )
        top_label_clean = res["labels"][0]
        return self.clean_to_raw.get(top_label_clean, top_label_clean)

    def predict_batch(self, texts: List[str]) -> List[str]:
        preds = []
        for text in texts:
            preds.append(self.predict_single(text))
        return preds


class FlanT5ZeroShotModel(BaseBenchmarkClassifier):
    """
    Zero-Shot Generative Foundation Model:
    google/flan-t5-base seq2seq text-to-text generative model.
    Evaluates zero-shot with 0 training rows using prompt-based generative classification.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        target_classes: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        self.name = "FLAN-T5-Base Zero-Shot"
        self.model_type = "Generative Foundation Model (Zero-Shot)"
        self.supervised_samples_needed = 0
        self.model_name = model_name

        if target_classes is None:
            target_classes = [
                "card_arrival",
                "change_pin",
                "transfer_fee_charged",
                "automatic_top_up",
                "lost_or_stolen_card",
            ]
        self.target_classes = target_classes
        self.classes_str = ", ".join(target_classes)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, dtype=dtype
        ).to(self.device)

    def fit(self, texts: List[str], labels: List[str]) -> float:
        # Zero-shot generative model requires 0 training data
        return 0.0

    def predict_single(self, text: str) -> str:
        prompt = (
            f"Classify the customer banking request into exactly one of these categories: [{self.classes_str}].\n"
            f"Request: {text}\n"
            f"Category:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
            )
        pred_str = (
            self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            .strip()
            .lower()
            .replace(" ", "_")
        )

        for c in self.target_classes:
            if c in pred_str or pred_str in c:
                return c
        return pred_str

    def predict_batch(self, texts: List[str]) -> List[str]:
        return [self.predict_single(t) for t in texts]


def _make_clm_head(
    width: int,
    depth: int = 2,
    proj: int = 512,
    activation: str = "gelu",
    layernorm: bool = False,
    residual: bool = False,
    hidden: int = 4096,
):
    """Constructs CLM projection head MLP: hidden -> width -> ... -> proj."""
    act_map = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}
    act_cls = act_map.get(activation, nn.GELU)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(hidden, width)
            self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 2))
            self.norms = nn.ModuleList(
                (nn.LayerNorm(width) if layernorm else nn.Identity())
                for _ in range(depth - 2)
            )
            self.out = nn.Linear(width, proj)
            self.act = act_cls()
            self.residual = residual

        def forward(self, x):
            x = self.act(self.inp(x))
            for lin, nrm in zip(self.hidden, self.norms):
                h = self.act(nrm(lin(x)))
                x = x + h if self.residual else h
            return self.out(x)

    return Head()


class Clm8bZeroShotModel(BaseBenchmarkClassifier):
    """
    Zero-Shot Contrastive Foundation Classifier:
    Stanford/NVIDIA CLM-8B (Contrastive-LM/CLM-v0.1-8B) with 4-bit NF4 quantized Qwen3-8B encoder.
    Uses frozen Qwen3-8B + trained contrastive state/action projection heads.
    Pre-computes and caches action embeddings for sub-millisecond candidate scoring.
    """

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen3-8B",
        head_repo: str = "Contrastive-LM/CLM-v0.1-8B",
        head_filename: str = "CLM_v0.1-8B.pt",
        target_classes: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        self.name = "NVIDIA/Stanford CLM-8B (4-bit)"
        self.model_type = "Contrastive Foundation Model (Zero-Shot 4-bit)"
        self.supervised_samples_needed = 0
        self.base_model_name = base_model_name

        if target_classes is None:
            target_classes = [
                "card_arrival",
                "change_pin",
                "transfer_fee_charged",
                "automatic_top_up",
                "lost_or_stolen_card",
            ]
        self.target_classes = target_classes

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info(f"Loading CLM-8B projection head from HF '{head_repo}'...")
        ckpt_path = hf_hub_download(head_repo, head_filename)
        ck = torch.load(ckpt_path, map_location="cpu")
        cfg = dict(ck["cfg"])
        head_kwargs = dict(
            width=cfg["width"],
            depth=cfg["depth"],
            proj=ck.get("projection_dim", cfg.get("projection_dim", 512)),
            activation=cfg.get("activation", "gelu"),
            layernorm=cfg.get("layernorm", False),
            residual=cfg.get("residual", False),
            hidden=cfg.get("hidden_size", 4096),
        )
        self.state_head = _make_clm_head(**head_kwargs)
        self.action_head = _make_clm_head(**head_kwargs)
        self.state_head.load_state_dict(ck["state_head"])
        self.action_head.load_state_dict(ck["action_head"])
        self.state_head.eval().to(self.device)
        self.action_head.eval().to(self.device)
        self.logit_scale = float(torch.as_tensor(ck["logit_scale"]).float().exp().clamp(max=100.0))

        logger.info(
            f"Loading base encoder '{base_model_name}' in 4-bit NF4 quantization on {self.device}..."
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.device == "cuda" or (isinstance(self.device, str) and self.device.startswith("cuda")):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.encoder = AutoModel.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            self.encoder = AutoModel.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
            ).to(self.device)
        self.encoder.eval()

        # Cache action projections for all candidate classes
        self._precompute_action_cache()

    def _get_embedding(self, text: str) -> torch.Tensor:
        """Extracts last-token pooled L2-normalized embedding (dim 4096)."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.encoder.device)
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            hidden_states = outputs.last_hidden_state
            last_token_idx = inputs.attention_mask.sum(dim=1) - 1
            emb = hidden_states[torch.arange(inputs.attention_mask.shape[0]), last_token_idx]
            emb = F.normalize(emb.float(), p=2, dim=-1)
        return emb

    def _precompute_action_cache(self) -> None:
        """Precomputes and caches action embeddings for all classes in projection space."""
        projs = []
        for c in self.target_classes:
            phrase = c.replace("_", " ")
            emb = self._get_embedding(phrase)
            with torch.no_grad():
                proj = F.normalize(self.action_head(emb.to(self.device)), p=2, dim=-1)
            projs.append(proj)
        self.cached_action_projs = torch.cat(projs, dim=0)

    def fit(self, texts: List[str], labels: List[str]) -> float:
        # Zero-shot model requires 0 training samples
        return 0.0

    def predict_single(self, text: str) -> str:
        # State layout: context first, question prompt last
        query = f"{text}\n\nWhich category best describes this customer banking request?"
        emb = self._get_embedding(query)
        with torch.no_grad():
            state_proj = F.normalize(self.state_head(emb.to(self.device)), p=2, dim=-1)
            scores = (state_proj @ self.cached_action_projs.T).squeeze(0)
            best_idx = torch.argmax(scores).item()
        return self.target_classes[best_idx]

    def predict_batch(self, texts: List[str]) -> List[str]:
        return [self.predict_single(t) for t in texts]


