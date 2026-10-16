"""Local multilingual document embeddings; no model downloads at runtime."""

from pathlib import Path

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
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Cannot embed empty document text.")
            tokens = self.model.tokenizer.encode(text, add_special_tokens=False)
            pieces = [tokens[start:start + 480] for start in range(0, len(tokens), 480)]
            chunks = ["query: " + self.model.tokenizer.decode(piece) for piece in pieces]
            embeddings = self.model.encode(
                chunks, batch_size=self.batch_size, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            )
            pooled = np.average(embeddings, axis=0, weights=[len(piece) for piece in pieces])
            vectors.append(pooled / np.linalg.norm(pooled))
        return np.asarray(vectors)
