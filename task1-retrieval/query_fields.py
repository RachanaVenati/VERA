"""BM25 uses query; reranking prefers description and falls back to query.

Descriptions come from original_query, matching the official input schema.
Narrative is never used.
"""


def retrieval_text(query) -> str:
    """Return the short query, without description or narrative expansion."""
    text = query.query
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Query {query.query_id} has no non-empty query field.")
    return text.strip()


def reranking_field(query) -> str:
    """Identify the effective reranking input for traces and candidate records."""
    metadata = getattr(query, "original_query", None)
    text = metadata.get("description") if isinstance(metadata, dict) else None
    return "description" if isinstance(text, str) and text.strip() else "query"


def reranking_text(query) -> str:
    """Use a nonempty description, otherwise the validated short query."""
    if reranking_field(query) == "description":
        return query.original_query["description"].strip()
    return retrieval_text(query)
