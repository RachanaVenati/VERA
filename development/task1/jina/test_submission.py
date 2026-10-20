"""Exercise complete offline entry point, persistent reuse and event tracing."""
import gzip
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'task1-jina'))
from event_logging import EventTrace
from jina_retrieval import execute, build_index
from jina_vector_store import VectorWriter


class Model:
    tokenizer=SimpleNamespace(encode=lambda text, **kw:text.split(),
                              decode=lambda ids, **kw:' '.join(ids))
    def encode(self,texts,**kwargs):
        assert kwargs['task']=='retrieval' and kwargs['prompt_name'] in ('query','document')
        return np.array([[text.count('alpha'),text.count('beta')] for text in texts],dtype=float)


class SubmissionTest(unittest.TestCase):
    def test_query_only_input_end_to_end_and_index_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'queries.jsonl').write_text(json.dumps({'qid':'q','query':'alpha'})+'\n')
            (root/'documents.jsonl').write_text(''.join(json.dumps(d)+'\n' for d in [
                {'doc_id':'a','text':'alpha alpha'}, {'doc_id':'b','text':'beta beta'}]))
            for name,reuse in [('first',None),('reused',root/'first/index')]:
                out=root/name;out.mkdir()
                trace=EventTrace(out/'run-trace.jsonl.log.gz','test')
                parent=trace.emit('__index__','input')
                args=SimpleNamespace(dataset=root,output=out,model=root,batch_size=2,buffer_size=4,reuse_index=reuse)
                execute(args,trace,parent,loader=lambda _:Model());trace.close()
                with gzip.open(out/'run.txt.gz','rt') as f:
                    self.assertEqual([r.split()[2] for r in f],['a','b'])
                with gzip.open(out/'run-trace.jsonl.log.gz','rt') as f:
                    events=[json.loads(r) for r in f]
                self.assertTrue(all(len(e)==11 for e in events))
                self.assertTrue(any(e['event_type']=='model_call' for e in events))
                self.assertTrue(any(e['case_id']=='q' and e['event_type']=='decision' for e in events))
            # Source changes must invalidate the reusable index.
            (root/'documents.jsonl').write_text(json.dumps({'doc_id':'a','text':'alpha beta'})+'\n')
            out=root/'bad';out.mkdir();trace=EventTrace(out/'trace.gz','test')
            args.output=out
            with self.assertRaisesRegex(ValueError,'does not match'):
                execute(args,trace,None,loader=lambda _:Model())
            trace.close()

    def test_long_document_preserves_all_tokens_and_bounds_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            writer=VectorWriter(directory);calls=[]
            def encode(texts,prompt,metadata):
                calls.extend((len(t.split()),m) for t,m in zip(texts,metadata))
                return np.array([[1,0]]*len(texts))
            counts=build_index([{'doc_id':'long','text':' '.join(['alpha']*1029)}],writer,
                               Model.tokenizer,encode,batch_size=2,buffer_size=2)
            writer.close({})
            self.assertEqual(len(calls),3)
            self.assertLessEqual(max(length for length,_ in calls),480)
            spans=sorted((m['start_token'],m['end_token']) for _,m in calls)
            self.assertEqual(spans,[(0,480),(416,896),(832,1029)])
            self.assertEqual(counts['documents_seen'],1)
