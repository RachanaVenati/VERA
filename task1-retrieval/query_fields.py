"""Keep the inputs for lexical retrieval and semantic reranking separate.

BM25 uses the short `query`. The cross-encoder uses `description`.
Descriptions are read from original_query, matching the official input schema.
Neither stage uses the narrative or silently substitutes another field.
"""


def retrieval_text(query) -> str:
    """Return the short query, without description or narrative expansion."""
    text = query.query
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Query {query.query_id} has no non-empty query field.")
    return text.strip()


def reranking_text(query) -> str:
    """Return the description; fail explicitly when reranking lacks its input."""
    metadata = query.original_query
    text = metadata.get("description") if isinstance(metadata, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Query {query.query_id} has no non-empty description field.")
    return text.strip()
