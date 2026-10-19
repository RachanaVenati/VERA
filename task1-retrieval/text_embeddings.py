"""Local multilingual document embeddings; no model downloads at runtime."""

from pathlib import Path
import time

import numpy as np

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


class DocumentEmbedder:
    """Embed all document chunks, then pool them into one document vector.

E5 uses the same 'query: ' prefix for both sides of symmetric similarity.
Chunks contain at most 480 content tokens, leaving room under its 512-token
limit for the prefix and special tokens. Every chunk is included. Weighted
mean pooling is an approximation to document similarity, not duplicate proof.
"""

    def __init__(self, model_path: str, batch_size: int = 16):
        if not Path(model_path).is_dir():
            raise ValueError(f"Local embedding model not found: {model_path}")
        if batch_size < 1:
            raise ValueError("Embedding batch size must be positive.")
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(4)
        self.model = SentenceTransformer(model_path, device="cpu", local_files_only=True)
        self.model.max_seq_length = 512
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        records = {}
        for record in getattr(self, "records", []):
            records.setdefault(record["text"], record)
        self.source_events = {}
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Cannot embed empty document text.")
            tokens = self.model.tokenizer.encode(text, add_special_tokens=False)
            pieces = [tokens[start:start + 480] for start in range(0, len(tokens), 480)]
            if not pieces:
                raise ValueError("Document contains no embedding tokens.")
            chunks = ["query: " + self.model.tokenizer.decode(piece) for piece in pieces]
            trace = getattr(self, "trace", None)
            record = records.get(text)
            started = time.monotonic()
            inputs = {"stage": "semantic_embedding", "document_id": record["doc_id"] if record else None,
                      "chunks": chunks, "batch_size": self.batch_size, "revision": MODEL_REVISION}
            try:
                embeddings = self.model.encode(
                    chunks, batch_size=self.batch_size, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False,
                )
                pooled = np.average(embeddings, axis=0, weights=[len(piece) for piece in pieces])
                norm = np.linalg.norm(pooled)
                if not np.isfinite(pooled).all() or norm == 0:
                    raise ValueError("Invalid pooled document embedding.")
                pooled = pooled / norm
                if trace and record:
                    event = trace.emit(record["qid"], "model_call", self.parents[str(record["qid"])],
                                       inputs=inputs, model=MODEL_ID, output={
                                           "chunk_embeddings": np.asarray(embeddings).tolist(),
                                           "document_embedding": pooled.tolist(),
                                           "seconds": time.monotonic() - started})
                    self.source_events[text] = {"case_id": str(record["qid"]), "event_id": event}
                vectors.append(pooled)
            except Exception as error:
                if trace and record:
                    trace.emit(record["qid"], "model_call", self.parents[str(record["qid"])],
                               inputs=inputs, model=MODEL_ID, status="error", error=str(error))
                raise
        return np.asarray(vectors)
