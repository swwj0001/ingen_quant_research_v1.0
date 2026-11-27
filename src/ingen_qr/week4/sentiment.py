"""Finance dictionary, sentence-level FinBERT and frozen sentence embeddings."""

from importlib import import_module
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from ingen_qr.week4.filings import WORD, sentences

LM_FIELDS = ["positive", "negative", "uncertainty", "litigious"]


def dictionary(path: Path) -> dict[str, set[str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Loughran–McDonald master dictionary not found at {path}. "
            "Download the CSV and place it there or change 'dictionary' in configs/week4_text.yaml."
        )
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame.columns = frame.columns.str.strip().str.lower()
    missing = {"word", *LM_FIELDS} - set(frame.columns)
    if missing:
        raise ValueError(
            f"Missing Loughran–McDonald columns in {path}: {sorted(missing)}"
        )
    words = frame["word"].astype(str).str.upper().str.strip()
    return {
        category: set(words.loc[pd.to_numeric(frame[category], errors="coerce").gt(0)])
        for category in LM_FIELDS
    }


def lm_scores(text: str, lexicon: dict[str, set[str]]) -> dict[str, float]:
    tokens = WORD.findall(text.upper())
    if not tokens:
        return dict.fromkeys(["lm_tone", "lm_uncertainty", "lm_litigious"], np.nan)
    counts = {
        category: sum(t in lexicon[category] for t in tokens) for category in LM_FIELDS
    }
    return {
        "lm_tone": (counts["positive"] - counts["negative"]) / len(tokens),
        "lm_uncertainty": counts["uncertainty"] / len(tokens),
        "lm_litigious": counts["litigious"] / len(tokens),
    }


def sample_sentences(lines: list[str], maximum: int) -> list[str]:
    if maximum < 1:
        raise ValueError("nlp_max_sentences_per_document must be positive.")
    if len(lines) <= maximum:
        return lines
    positions = sorted(Random(1).sample(range(len(lines)), k=maximum))
    return [lines[position] for position in positions]


GPU_BATCH_SIZE = 128
CPU_BATCH_SIZE = 32


def resolve_device(requested: Any, torch: Any) -> tuple[str, str]:
    """Map 'auto' to CUDA when PyTorch can see a GPU; fail loudly on a missing GPU."""
    value = str(requested).strip().lower()
    available = bool(torch.cuda.is_available())
    if value == "auto":
        value = "cuda" if available else "cpu"
    if value.startswith("cuda"):
        if not available:
            raise RuntimeError(
                f"nlp_device is {requested!r} but PyTorch {torch.__version__} cannot see a CUDA GPU. "
                "Install the CUDA build of torch or set nlp_device to auto or cpu."
            )
        index = int(value.split(":", 1)[1]) if ":" in value else 0
        return value, str(torch.cuda.get_device_name(index))
    if value != "cpu":
        raise ValueError("nlp_device must be auto, cpu, cuda or cuda:<index>.")
    return "cpu", "CPU"


def resolve_batch(requested: Any, device: str) -> int:
    if str(requested).strip().lower() == "auto":
        return GPU_BATCH_SIZE if device.startswith("cuda") else CPU_BATCH_SIZE
    batch = int(requested)
    if batch < 1:
        raise ValueError("nlp_batch_size must be positive or auto.")
    return batch


def nlp_runtime(settings: dict[str, Any]) -> tuple[str, str, int]:
    """Resolve the actual NLP device and batch size for this machine."""
    try:
        torch = import_module("torch")
    except ImportError as error:
        raise RuntimeError(
            "Install the Week 4 optional text dependency group and run with --group week4-text."
        ) from error
    device, name = resolve_device(settings["nlp_device"], torch)
    return device, name, resolve_batch(settings["nlp_batch_size"], device)


class Scorers:
    def __init__(self, settings: dict[str, Any], lexicon: dict[str, set[str]]) -> None:
        self.settings = settings
        self.lexicon = lexicon
        self.max_sentences = int(settings["nlp_max_sentences_per_document"])
        if self.max_sentences < 1:
            raise ValueError("nlp_max_sentences_per_document must be positive.")
        try:
            self.torch = import_module("torch")
            transformers = import_module("transformers")
            sentence_transformers = import_module("sentence_transformers")
        except ImportError as error:
            raise RuntimeError(
                "Install the Week 4 optional text dependency group and run with --group week4-text."
            ) from error
        self.device, self.device_name = resolve_device(
            settings["nlp_device"], self.torch
        )
        self.batch = resolve_batch(settings["nlp_batch_size"], self.device)
        started = perf_counter()
        model, rev = settings["finbert_model"], settings["finbert_revision"]
        self.fin_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model, revision=rev
        )
        self.finbert = (
            transformers.AutoModelForSequenceClassification.from_pretrained(
                model, revision=rev
            )
            .to(self.device)
            .eval()
        )
        self.labels = {
            v.lower(): int(k) for k, v in self.finbert.config.id2label.items()
        }
        if set(self.labels) != {"positive", "negative", "neutral"}:
            raise ValueError(f"Unexpected FinBERT labels: {self.labels}")
        self.finbert_setup_seconds = perf_counter() - started
        started = perf_counter()
        self.encoder = sentence_transformers.SentenceTransformer(
            settings["embedding_model"],
            revision=settings["embedding_revision"],
            device=self.device,
        )
        self.embedding_setup_seconds = perf_counter() - started

    def finbert_scores(self, lines: list[str]) -> dict[str, float]:
        tokenizer = self.fin_tokenizer
        limit = (
            self.finbert.config.max_position_embeddings
            - tokenizer.num_special_tokens_to_add()
        )
        encoded = tokenizer(lines, add_special_tokens=False, truncation=False)[
            "input_ids"
        ]
        parts, owners, weights = [], [], []
        for ix, ids in enumerate(encoded):
            for start in range(0, len(ids), limit):
                tokens = ids[start : start + limit]
                parts.append(tokenizer.prepare_for_model(tokens, truncation=False))
                owners.append(ix)
                weights.append(len(tokens))
        columns = ["positive", "negative", "neutral"]
        if not parts:
            return {
                "finbert_tone": np.nan,
                **{f"finbert_{key}": np.nan for key in columns},
            }
        probabilities = []
        with self.torch.inference_mode():
            for start in range(0, len(parts), self.batch):
                batch = tokenizer.pad(
                    parts[start : start + self.batch], return_tensors="pt"
                )
                predictions = self.finbert(**batch.to(self.device)).logits.softmax(
                    dim=-1
                )
                probabilities.append(predictions.cpu().numpy())
        matrix = np.concatenate(probabilities)
        totals = np.zeros((len(lines), matrix.shape[1]))
        counts = np.bincount(owners, weights=weights, minlength=len(lines))
        np.add.at(totals, owners, matrix * np.asarray(weights)[:, None])
        observed = counts > 0
        mean = (totals[observed] / counts[observed, None]).mean(axis=0)
        return {
            "finbert_tone": float(
                mean[self.labels["positive"]] - mean[self.labels["negative"]]
            ),
            **{f"finbert_{name}": float(mean[self.labels[name]]) for name in columns},
        }

    def embeddings(self, lines: list[str]) -> list[float] | None:
        tokenizer = self.encoder.tokenizer
        limit = self.encoder.max_seq_length - tokenizer.num_special_tokens_to_add()
        encoded = tokenizer(lines, add_special_tokens=False, truncation=False)[
            "input_ids"
        ]
        chunks, weights, sentence_ids = [], [], []
        for ix, ids in enumerate(encoded):
            if len(ids) <= limit:
                if ids:
                    chunks.append(lines[ix])
                    weights.append(len(ids))
                    sentence_ids.append(ix)
            else:
                for start in range(0, len(ids), limit):
                    tokens = ids[start : start + limit]
                    chunks.append(tokenizer.decode(tokens, skip_special_tokens=True))
                    weights.append(len(tokens))
                    sentence_ids.append(ix)
        if not chunks:
            return None
        matrix = self.encoder.encode(
            chunks,
            batch_size=self.batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        totals = np.zeros((len(lines), matrix.shape[1]))
        counts = np.bincount(sentence_ids, weights=weights, minlength=len(lines))
        np.add.at(totals, sentence_ids, matrix * np.asarray(weights)[:, None])
        observed = counts > 0
        vector = (totals[observed] / counts[observed, None]).mean(axis=0)
        norm = np.linalg.norm(vector)
        return (vector / norm).tolist() if norm > 0 else None

    def score(self, text: str) -> dict[str, Any]:
        lines = sentences(text)
        sampled = sample_sentences(lines, self.max_sentences)
        start = perf_counter()
        lm = lm_scores(text, self.lexicon)
        lm_time = perf_counter() - start
        start = perf_counter()
        fin = self.finbert_scores(sampled)
        fin_time = perf_counter() - start
        start = perf_counter()
        embedding = self.embeddings(sampled)
        embedding_time = perf_counter() - start
        return {
            **lm,
            **fin,
            "embedding": embedding,
            "document_sentences": len(lines),
            "nlp_sampled_sentences": len(sampled),
            "lm_seconds": lm_time,
            "finbert_seconds": fin_time,
            "embedding_seconds": embedding_time,
        }
