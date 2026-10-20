#!/usr/bin/env python3
"""TIRA entry point: query BM25 -> description reranking -> complete TREC run."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import shutil
import tempfile
import time

import click
from tira.third_party_integrations import ir_datasets
from baseline import create_index, detect_query_language, retrieve
from candidates import prepare_candidates
from cross_encoder import CrossEncoderScorer, MODEL_ID, MODEL_REVISION, RerankingBudgetExceeded
from event_logging import EventTrace
from query_fields import retrieval_text, reranking_text, reranking_field
from reranking import rerank_candidates
from group_ranking import order_candidates, STRATEGIES


def merge_rankings(bm25_run, reranked):
    """Reranked prefix, then untouched BM25 tail; reciprocal-rank TREC scores."""
    prefixes, seen = defaultdict(list), set()
    for record in reranked:
        key = (str(record["qid"]), str(record["doc_id"]))
        if key in seen:
            raise ValueError(f"Duplicate reranked ID: {key}")
        seen.add(key)
        prefixes[key[0]].append(record)
    output, covered = [], set()
    for qid, rows in bm25_run.groupby("qid", sort=True):
        qid = str(qid)
        ordered = rows.sort_values("rank", kind="stable")
        bm25 = {str(row.docno): row for row in ordered.itertuples(index=False)}
        if len(bm25) != len(ordered):
            raise ValueError(f"Repeated BM25 document ID for query {qid}.")
        prefix = sorted(prefixes.get(qid, []), key=lambda row: row["rerank_rank"])
        prefix_ids = {str(row["doc_id"]) for row in prefix}
        if prefix_ids != set(list(bm25)[:len(prefix)]):
            raise ValueError(f"Reranked prefix must cover the original top candidates for query {qid}.")
        sequence = [str(row["doc_id"]) for row in prefix]
        sequence.extend(doc_id for doc_id in bm25 if doc_id not in prefix_ids)
        for rank, doc_id in enumerate(sequence, 1):
            output.append({"qid": qid, "doc_id": doc_id, "rank": rank, "score": 1.0/rank})
            covered.add((qid, doc_id))
    if not seen.issubset(covered):
        raise ValueError("Reranked results include IDs outside the BM25 run.")
    return output


def write_run(records, path):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for record in records:
            qid, doc_id = record["qid"], record["doc_id"]
            if not qid or not doc_id or any(c.isspace() for c in qid + doc_id):
                raise ValueError("Invalid TREC query/document ID.")
            stream.write(f"{qid} Q0 {doc_id} {record['rank']} {record['score']:.12g} vera\n")


class AuditedScorer:
    """Associate each local model batch with its query/candidate observation."""
    def __init__(self, scorer, candidates, trace, parents, case_budget_seconds=240):
        self.scorer, self.trace, self.parents = scorer, trace, parents
        self.records = {}
        for record in sorted(candidates, key=lambda r: (r["qid"], r["bm25_rank"], r["doc_id"])):
            self.records.setdefault((record["description"], record["text"]), record)
        self.case_seconds = defaultdict(float)
        self.score_events = {}
        self.last_qid = None
        self.case_budget_seconds = case_budget_seconds
        self.limited_queries = set()

    def score(self, description, text):
        record = self.records[(description, text)]
        if record["qid"] != self.last_qid:
            if self.last_qid is not None:
                click.echo(f"Finished query {self.last_qid}: {self.case_seconds[self.last_qid]:.1f}s reranking", err=True)
            self.last_qid = record["qid"]
            self.scorer.deadline = time.monotonic() + self.case_budget_seconds
            click.echo(f"Reranking query {self.last_qid}", err=True)
        self.scorer.trace_context = {"trace": self.trace, "qid": record["qid"],
                                     "doc_id": record["doc_id"], "description": description,
                                     "parent": self.parents[record["qid"]]}
        started = time.monotonic()
        try:
            result = self.scorer.score(description, text)
            self.score_events[(description, text)] = self.scorer.trace_context.get("last_event")
            return result
        except RerankingBudgetExceeded:
            self.limited_queries.add(record["qid"])
            self.trace.emit(record["qid"], "observation", self.parents[record["qid"]], output={
                "stage": "reranking", "stopped_reason": "time_budget",
                "unscored_document_id": record["doc_id"], "policy": "retain remaining BM25 tail"})
            raise
        except Exception as error:
            self.trace.emit(record["qid"], "error", self.parents[record["qid"]],
                            inputs={"description": description, "document_id": record["doc_id"]},
                            status="error", error=str(error))
            raise
        finally:
            self.case_seconds[record["qid"]] += time.monotonic() - started


@click.command()
@click.option("--dataset", envvar="inputDataset", required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", envvar="outputDir", required=True,
              type=click.Path(file_okay=False, path_type=Path))
@click.option("--candidate-k", default=50, show_default=True, type=click.IntRange(1, 1000))
@click.option("--mode", default="rerank", type=click.Choice(["rerank", "bm25"]), show_default=True)
@click.option("--batch-size", default=16, type=click.IntRange(1), show_default=True)
@click.option("--case-budget-seconds", default=240.0, type=click.FloatRange(min=10), show_default=True)
@click.option("--semantic-mode", default="off", type=click.Choice(["off", "audit"]), show_default=True)
@click.option("--semantic-model-path", default="/models/multilingual-e5-small", show_default=True,
              type=click.Path(file_okay=False, path_type=Path))
@click.option("--semantic-threshold", default=0.95, type=click.FloatRange(min=0, max=1, min_open=True))
@click.option("--semantic-review-floor", default=0.90, type=click.FloatRange(min=0, max=1, min_open=True))
@click.option("--ranking-strategy", default="cross-encoder", type=click.Choice(STRATEGIES), show_default=True)
@click.option("--mmr-lambda", default=0.8, type=click.FloatRange(min=0, max=1), show_default=True)
def main(dataset, output, candidate_k, mode, batch_size, case_budget_seconds,
         semantic_mode, semantic_model_path, semantic_threshold, semantic_review_floor,
         ranking_strategy, mmr_lambda):
    if ranking_strategy != "cross-encoder" and (mode != "rerank" or semantic_mode != "audit"):
        raise click.ClickException("Group ranking requires --mode rerank --semantic-mode audit.")
    if semantic_mode == "audit":
        if mode != "rerank":
            raise click.ClickException("Semantic auditing requires --mode rerank.")
        if not semantic_model_path.is_dir():
            raise click.ClickException("Semantic model is missing. Build with --build-arg INCLUDE_SEMANTIC_MODEL=1.")
        if semantic_review_floor > semantic_threshold:
            raise click.ClickException("Semantic review floor cannot exceed the grouping threshold.")
    if output.exists() and any(output.iterdir()):
        raise click.ClickException("Output directory must be empty.")
    output.mkdir(parents=True, exist_ok=True)
    trace = EventTrace(output / "run-trace.jsonl.log.gz", MODEL_ID if mode == "rerank" else "-")
    started = time.monotonic()
    try:
        data = ir_datasets.load(str(dataset))
        language = detect_query_language(data)
        inputs = {}
        reranking_fields = {}
        for query in data.queries_iter():
            description = reranking_text(query) if mode == "rerank" else None
            if mode == "rerank":
                reranking_fields[str(query.query_id)] = reranking_field(query)
            inputs[str(query.query_id)] = trace.emit(query.query_id, "input", inputs={
                "query": retrieval_text(query), "description": description, "language": language,
                "reranking_field": reranking_fields.get(str(query.query_id))})
        with tempfile.TemporaryDirectory(prefix="vera-") as temporary:
            work = Path(temporary)
            index = create_index(data, language, work, index_directory=work / "index")
            try:
                bm25 = retrieve(data, index, language, "BM25", work)
            finally:
                index.close()
            candidates = prepare_candidates(data, bm25, candidate_k, dataset) if mode == "rerank" else []
            parents = {}
            for qid, event in inputs.items():
                rows = bm25[bm25.qid.astype(str) == qid].sort_values("rank")
                parents[qid] = trace.emit(qid, "observation", event, output={
                    "stage": "BM25", "retrieved_document_ids": list(rows.docno.astype(str)),
                    "rerank_candidate_ids": [r["doc_id"] for r in candidates if r["qid"] == qid]})
            groups, semantic_summary, similarities = None, {"mode": "off"}, None
            if semantic_mode == "audit":
                from semantic_audit import run_semantic_audit
                from text_embeddings import DocumentEmbedder
                embedder = DocumentEmbedder(str(semantic_model_path), batch_size)
                embedder.trace, embedder.parents = trace, parents
                embedder.records = candidates
                groups, semantic_summary, similarities = run_semantic_audit(
                    candidates, embedder, output, trace, parents,
                    semantic_threshold, semantic_review_floor)
                del embedder
            case_seconds, results, last_observations, limited_queries = {}, [], dict(parents), []
            if candidates:
                audited = AuditedScorer(CrossEncoderScorer(batch_size=batch_size), candidates, trace,
                                       parents, case_budget_seconds)
                results = rerank_candidates(candidates, audited, groups)
                results = order_candidates(results, groups, similarities, ranking_strategy, mmr_lambda)
                case_seconds = dict(audited.case_seconds)
                limited_queries = sorted(audited.limited_queries)
                with gzip.open(output / "reranking-scores.jsonl.gz", "wt", encoding="utf-8") as stream:
                    for row in results:
                        scores = {key: row[key] for key in (
                            "qid", "doc_id", "bm25_rank", "bm25_score", "rerank_rank", "rerank_score",
                            "chunk_count", "best_chunk_index", "score_source_doc_id")}
                        scores.update({key: row[key] for key in ("cross_encoder_rank", "ranking_strategy")})
                        if "selection_score" in row:
                            scores["selection_score"] = row["selection_score"]
                        stream.write(json.dumps(scores, allow_nan=False) + "\n")
                        source = audited.records[(row["description"], row["text"])]
                        event = audited.score_events[(row["description"], row["text"])]
                        scores["source_model_event_id"] = event
                        parent = event if source["qid"] == row["qid"] else parents[row["qid"]]
                        last_observations[row["qid"]] = trace.emit(
                            row["qid"], "observation", parent, output=scores)
            final = merge_rankings(bm25, results)
            write_run(final, output / "run.txt.gz")
            for qid in inputs:
                answer = [row for row in final if row["qid"] == qid]
                evidence = [{key: row[key] for key in (
                    "doc_id", "rerank_score", "best_chunk_index", "score_source_doc_id")}
                    for row in results if row["qid"] == qid and row["rerank_rank"] <= 10]
                trace.emit(qid, "decision", last_observations[qid], output={
                    "decision": "rank_documents", "ranking_strategy": ranking_strategy,
                    "mmr_lambda": mmr_lambda if ranking_strategy == "mmr" else None,
                    "answer": answer, "evidence": evidence})
            for name in ("index-ir-metadata.yml", "retrieval-ir-metadata.yml"):
                shutil.copy2(work / name, output / name)
        summary = {"mode": mode, "language": language, "candidate_k": candidate_k,
                   "model": MODEL_ID if mode == "rerank" else "-",
                   "revision": MODEL_REVISION if mode == "rerank" else None,
                   "retrieval_field": "query", "reranking_field": (next(iter(set(reranking_fields.values())))
                                       if len(set(reranking_fields.values())) == 1
                                       else "mixed" if reranking_fields else None),
                   "reranking_policy": "nonempty description, otherwise query",
                   "reranking_fields_by_query": reranking_fields,
                   "semantic_audit": semantic_summary,
                   "ranking_strategy": ranking_strategy, "mmr_lambda": mmr_lambda if ranking_strategy == "mmr" else None,
                   "deduplication": "exact description/text score reuse; all document IDs preserved",
                   "final_score": "reciprocal rank; BM25 tail follows reranked prefix",
                   "queries": len(inputs), "results": len(final),
                   "case_budget_seconds": case_budget_seconds,
                   "budget_limited_queries": limited_queries,
                   "case_reranking_seconds": case_seconds, "seconds": time.monotonic() - started}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        click.echo(json.dumps(summary, indent=2))
    except Exception as error:
        trace.emit("__run__", "error", status="error", error=str(error))
        raise
    finally:
        trace.close()


if __name__ == "__main__":
    main()
