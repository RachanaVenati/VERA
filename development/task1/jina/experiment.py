"""Build a persistent Jina Nano index, then exact dense-only document search."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

from vector_store import VectorWriter, search

MODEL='jinaai/jina-embeddings-v5-text-nano'
REVISION='8a7f00aac812071b69403df470f1038ec85f8925'


def records(path):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt') as stream:
        for line in stream:
            yield json.loads(line)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=Path('/models/jina-nano'))
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be empty')
    args.output.mkdir(parents=True,exist_ok=True)
    import torch
    from transformers import AutoModel
    torch.set_num_threads(4)
    started=time.monotonic()
    model=AutoModel.from_pretrained(str(args.model),trust_remote_code=True,dtype=torch.float32,
                                    local_files_only=True).to('cpu').eval()
    tokenizer=model.tokenizer
    queries=list(records(args.dataset/'queries.jsonl'))
    if not queries or len({str(q['qid']) for q in queries})!=len(queries):
        raise ValueError('Missing or duplicate queries')
    def encode(texts,prompt):
        prefix='Query: ' if prompt=='query' else 'Document: '
        if any(len(tokenizer.encode(prefix+t,add_special_tokens=True,verbose=False))>512 for t in texts):
            raise ValueError('Input exceeds configured 512-token comparison window')
        with torch.inference_mode():
            return model.encode(texts=texts,task='retrieval',prompt_name=prompt,max_length=512).detach().cpu().float().numpy()
    query_vectors=encode([q['query'].strip() for q in queries],'query')
    writer=VectorWriter(args.output/'index')
    texts,metadata=[],[]
    def flush():
        if texts:
            writer.add(encode(texts,'document'),metadata)
            texts.clear();metadata.clear()
    seen=set()
    path=args.dataset/'documents.jsonl.gz'
    if not path.exists():path=args.dataset/'documents.jsonl'
    for document in records(path):
        doc_id=str(document['doc_id'])
        if doc_id in seen:raise ValueError('Duplicate document ID')
        seen.add(doc_id)
        text=document['text']
        if not isinstance(text,str) or not text.strip():continue
        ids=tokenizer.encode(text,add_special_tokens=False,truncation=False,verbose=False)
        start=0
        while start<len(ids):
            end=min(start+480,len(ids))
            texts.append(tokenizer.decode(ids[start:end],skip_special_tokens=True))
            metadata.append({'doc_id':doc_id,'start_token':start,'end_token':end})
            if len(texts)==16:flush()
            if end==len(ids):break
            start=end-64
        if len(seen)%100==0:print(json.dumps({'documents_seen':len(seen),'chunks_written':writer.count}),flush=True)
    flush()
    manifest={'model':MODEL,'revision':REVISION,'task':'retrieval','chunk_tokens':480,'overlap':64,
              'dataset_documents_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    writer.close(manifest)
    index_seconds=time.monotonic()-started
    search_start=time.monotonic()
    scores=search(args.output/'index',query_vectors)
    with gzip.open(args.output/'run.txt.gz','wt') as stream:
        for i,q in enumerate(queries):
            ordered=sorted(scores,key=lambda d:(-float(scores[d][i]),d))[:1000]
            for rank,doc_id in enumerate(ordered,1):
                stream.write(f"{q['qid']} Q0 {doc_id} {rank} {scores[doc_id][i]:.9g} vera-jina-nano\n")
    summary={**manifest,'embedding_dimension':writer.dimension,'chunks':writer.count,'documents_seen':len(seen),
             'documents_embedded':len(scores),'query_field':'query','query_count':len(queries),
             'index_seconds_including_model_load_and_query_encoding':index_seconds,
             'search_seconds':time.monotonic()-search_start,'seconds':time.monotonic()-started,
             'aggregation':'max chunk cosine per document','storage':'index/embeddings.f32 + chunks.jsonl + manifest.json'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
