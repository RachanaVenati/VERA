"""Offline cross-encoder adapter: description/document chunks -> relevance."""

from pathlib import Path
import math

MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MODEL_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
DEFAULT_MODEL_PATH = "/models/mmarco-cross-encoder"


def pair_chunks(tokenizer, description: str, text: str,
                max_length: int = 512, overlap: int = 64) -> list[dict]:
    """Keep the full description and cover every document token.

Construct token pairs directly, avoiding decode/re-tokenize drift and implicit
truncation. The description repeats in each chunk; overlap spans boundaries.
"""
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Cross-encoder requires a non-empty description.")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Cross-encoder requires non-empty document text.")
    query_ids = tokenizer.encode(description, add_special_tokens=False, verbose=False)
    document_ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
    budget = max_length - len(query_ids) - tokenizer.num_special_tokens_to_add(pair=True)
    if budget <= 0:
        raise ValueError("Description leaves no document space; it is not silently truncated.")
    if overlap < 0 or overlap >= budget:
        raise ValueError("Chunk overlap must be non-negative and below the document token budget.")
    if not document_ids:
        raise ValueError("Document text contains no tokens.")
    chunks = []
    start = 0
    while start < len(document_ids):
        end = min(start + budget, len(document_ids))
        chunks.append(tokenizer.prepare_for_model(
            query_ids, pair_ids=document_ids[start:end], truncation=False,
            return_attention_mask=True,
            return_token_type_ids="token_type_ids" in tokenizer.model_input_names,
            verbose=False,
        ))
        if end == len(document_ids):
            break
        start = end - overlap
    return chunks


class CrossEncoderScorer:
    """CPU inference with a local model; max chunk logit is the document score."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, batch_size: int = 16,
                 max_length: int = 512, overlap: int = 64):
        if not Path(model_path).is_dir():
            raise ValueError(f"Local cross-encoder model not found: {model_path}")
        if batch_size < 1 or not 1 <= max_length <= 512 or not 0 <= overlap < max_length:
            raise ValueError("Invalid batch size, sequence length, or chunk overlap.")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(4)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True,
        ).to("cpu").eval()
        if self.model.config.num_labels != 1:
            raise ValueError("Cross-encoder must return a single relevance logit per pair.")
        self.batch_size, self.max_length, self.overlap = batch_size, max_length, overlap

    def _score_features(self, features: list[dict]) -> list[float]:
        scores = []
        with self.torch.inference_mode():
            for start in range(0, len(features), self.batch_size):
                batch = self.tokenizer.pad(features[start:start + self.batch_size],
                                           padding=True, return_tensors="pt")
                scores.extend(self.model(**batch).logits.reshape(-1).tolist())
        return scores

    def score(self, description: str, text: str) -> dict:
        features = pair_chunks(self.tokenizer, description, text, self.max_length, self.overlap)
        scores = self._score_features(features)
        if len(scores) != len(features) or not all(math.isfinite(s) for s in scores):
            raise ValueError("Cross-encoder returned invalid chunk scores.")
        best = max(range(len(scores)), key=lambda index: scores[index])
        return {"score": float(scores[best]), "chunk_count": len(scores),
                "best_chunk_index": best}
