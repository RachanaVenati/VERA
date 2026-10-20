FROM python:3.11-slim
WORKDIR /app
ENV HF_HOME=/opt/huggingface OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
COPY requirements.txt /app/
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('jinaai/jina-embeddings-v5-text-nano',revision='8a7f00aac812071b69403df470f1038ec85f8925',local_dir='/models/jina-nano',allow_patterns=['*.json','*.py','*.safetensors','*.model','*.txt'])"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN --network=none python -c "from transformers import AutoModel; import torch; torch.set_num_threads(4); m=AutoModel.from_pretrained('/models/jina-nano',trust_remote_code=True,dtype=torch.float32,local_files_only=True); assert m.encode(texts=['travel reimbursement'],task='retrieval',prompt_name='query',max_length=512).shape == (1,768); assert m.encode(texts=['Reisekostenerstattung'],task='retrieval',prompt_name='document',max_length=512).shape == (1,768)"
COPY jina_retrieval.py jina_vector_store.py event_logging.py submission.py /app/
ENTRYPOINT ["python", "/app/submission.py"]
