FROM ghcr.io/uniagent-webis-de/uniagent-pyterrier-baseline:dev-0.0.1@sha256:955bc071aefd3868e6d7292e4f02ee6579f781d66ed2e683354432b0c7de3d8e
WORKDIR /app
RUN python3 -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', revision='1427fd652930e4ba29e8149678df786c240d8825', local_dir='/models/mmarco-cross-encoder', allow_patterns=['*.json', '*.safetensors', '*.model', 'README.md'])"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COPY baseline.py candidates.py query_fields.py cross_encoder.py reranking.py event_logging.py submission.py /app/
COPY german-stopwords.txt german-stopwords.NOTICE /app/
ENTRYPOINT ["python3", "/app/submission.py"]
