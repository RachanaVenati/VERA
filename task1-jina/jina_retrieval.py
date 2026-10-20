"""Offline Jina Nano retrieval: reusable chunk index, exact cosine, original IDs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

from event_logging import EventTrace
from jina_vector_store import VectorWriter, search

MODEL = 'jinaai/jina-embeddings-v5-text-nano'
REVISION = '8a7f00aac812071b69403df470f1038ec85f8925'


def records(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def valid_id(value):
    value = str(value)
    if not value or any(c.isspace() for c in value):
        raise ValueError('Empty or whitespace-containing TREC ID')
    return value


def chunk_records(document, tokenizer):
    doc_id = valid_id(document['doc_id'])
    text = document['text']
    if not isinstance(text, str) or not text.strip():
        return
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
    start = 0
    while start < len(tokens):
        end = min(start + 480, len(tokens))
        chunk = tokenizer.decode(tokens[start:end], skip_special_tokens=True)
        yield chunk, {'doc_id': doc_id, 'start_token': start, 'end_token': end}
        if end == len(tokens):
            break
        start = end - 64


def build_index(documents, writer, tokenizer, encode, batch_size=16, buffer_size=128):
    """Bucket a bounded chunk buffer by length to reduce CPU padding work."""
    buffer, seen = [], set()
    skipped = 0
    def flush():
        buffer.sort(key=lambda item: item[1]['end_token'] - item[1]['start_token'])
        for start in range(0, len(buffer), batch_size):
            batch = buffer[start:start + batch_size]
            metadata = [m for _, m in batch]
            writer.add(encode([t for t, _ in batch], 'document', metadata), metadata)
        buffer.clear()
    for document in documents:
        doc_id = valid_id(document['doc_id'])
        if doc_id in seen:
            raise ValueError(f'Duplicate document ID: {doc_id}')
        seen.add(doc_id)
        count = 0
        for text, metadata in chunk_records(document, tokenizer):
            buffer.append((text, metadata))
            count += 1
            if len(buffer) >= buffer_size:
                flush()
        skipped += count == 0
        if len(seen) % 100 == 0:
            print(json.dumps({'documents_seen': len(seen), 'chunks_written': writer.count}), flush=True)
    flush()
    return {'documents_seen': len(seen), 'empty_documents_skipped': skipped}


def execute(args, trace, parent, loader=None):
    import numpy as np
    if loader is None:
        import torch
        from transformers import AutoModel
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        def loader(path):
            return AutoModel.from_pretrained(str(path), trust_remote_code=True, dtype=torch.float32,
                                             local_files_only=True).to('cpu').eval()
    started = time.monotonic()
    model = loader(args.model)
    tokenizer = model.tokenizer
    def encode(texts, prompt, metadata=None):
        prefix = 'Query: ' if prompt == 'query' else 'Document: '
        if any(len(tokenizer.encode(prefix + t, add_special_tokens=True, verbose=False)) > 512 for t in texts):
            raise ValueError('Input exceeds 512 tokens; refusing silent truncation')
        inputs = {'stage': prompt + '_embedding', 'texts': list(texts), 'chunks': metadata,
                  'task': 'retrieval', 'prompt_name': prompt, 'revision': REVISION}
        call_started = time.monotonic()
        try:
            vectors = model.encode(texts=texts, task='retrieval', prompt_name=prompt, max_length=512)
            if hasattr(vectors, 'detach'):
                vectors = vectors.detach().cpu().float().numpy()
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
                raise ValueError('Invalid model embedding output')
            trace.emit('__index__', 'model_call', parent, inputs=inputs,
                       output={'shape': list(vectors.shape), 'seconds': time.monotonic()-call_started})
            return vectors
        except Exception as error:
            trace.emit('__index__', 'model_call', parent, inputs=inputs, status='error', error=str(error))
            raise
    queries = list(records(args.dataset / 'queries.jsonl'))
    if not queries or len({valid_id(q['qid']) for q in queries}) != len(queries):
        raise ValueError('Missing or duplicate query IDs')
    for q in queries:
        if not isinstance(q.get('query'), str) or not q['query'].strip():
            raise ValueError('Missing nonempty query field')
    parents = {str(q['qid']): trace.emit(q['qid'], 'input', inputs={'query': q['query']}) for q in queries}
    query_vectors = np.concatenate([encode([q['query'].strip() for q in queries[start:start+args.batch_size]], 'query')
                                    for start in range(0, len(queries), args.batch_size)])
    path = args.dataset / 'documents.jsonl.gz'
    if not path.exists():
        path = args.dataset / 'documents.jsonl'
    manifest = {'model': MODEL, 'revision': REVISION, 'task': 'retrieval', 'chunk_tokens': 480,
                'overlap': 64, 'max_length': 512, 'dataset_documents_sha256': file_hash(path)}
    index = args.output / 'index'
    if args.reuse_index:
        saved = json.loads((args.reuse_index / 'manifest.json').read_text())
        if any(saved.get(k) != v for k, v in manifest.items()):
            raise ValueError('Saved index model, chunk settings, or corpus does not match')
        index = args.reuse_index
        counts = {'index_reused': True}
    else:
        writer = VectorWriter(index)
        try:
            counts = build_index(records(path), writer, tokenizer, encode, args.batch_size, args.buffer_size)
            writer.close(manifest)
        except Exception:
            writer.abort()
            raise
    index_seconds = time.monotonic()-started
    search_started = time.monotonic()
    scores = search(index, query_vectors)
    retrieval_seconds = time.monotonic()-search_started
    with gzip.open(args.output / 'run.txt.gz', 'wt', encoding='utf-8') as stream:
        for i, q in enumerate(queries):
            ordered = sorted(scores, key=lambda d: (-float(scores[d][i]), d))[:1000]
            trace.emit(q['qid'], 'decision', parents[str(q['qid'])], output={
                'decision': 'rank_documents', 'answer': ordered[:10],
                'evidence': [{'document_id': d, 'cosine': float(scores[d][i])} for d in ordered[:10]]})
            for rank, doc_id in enumerate(ordered, 1):
                stream.write(f"{q['qid']} Q0 {doc_id} {rank} {scores[doc_id][i]:.9g} vera-jina-nano\n")
    summary = {**manifest, **counts, 'documents_embedded': len(scores), 'query_count': len(queries),
               'query_field': 'query', 'batch_size': args.batch_size, 'buffer_size': args.buffer_size,
               'index_seconds_including_model_and_queries': index_seconds, 'search_seconds': retrieval_seconds,
               'seconds': time.monotonic()-started, 'aggregation': 'maximum chunk cosine per document',
               'cpu_threads': 4, 'index_directory': str(index), 'network': 'offline'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('/models/jina-nano'))
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--buffer-size', type=int, default=128)
    parser.add_argument('--reuse-index', type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.buffer_size < args.batch_size:
        parser.error('Require positive batch size and buffer >= batch size')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    trace = EventTrace(args.output / 'run-trace.jsonl.log.gz', MODEL)
    parent = trace.emit('__index__', 'input', inputs={'revision': REVISION, 'stage': 'shared_dense_index'})
    try:
        execute(args, trace, parent)
    except Exception as error:
        trace.emit('__index__', 'error', parent, status='error', error=str(error))
        raise
    finally:
        trace.close()
