FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('jinaai/jina-embeddings-v5-text-nano',revision='8a7f00aac812071b69403df470f1038ec85f8925',local_dir='/models/jina-nano',allow_patterns=['*.json','*.py','*.safetensors','*.model','*.txt'])"
RUN python -c "from transformers import AutoModel; import torch; m=AutoModel.from_pretrained('/models/jina-nano',trust_remote_code=True,dtype=torch.float32); print(m.encode(texts=['test retrieval'],task='retrieval',prompt_name='query',max_length=512).shape)"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COPY experiment.py vector_store.py /app/
ENTRYPOINT ["python","/app/experiment.py"]
