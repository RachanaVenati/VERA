FROM ghcr.io/uniagent-webis-de/uniagent-pyterrier-baseline:dev-0.0.1@sha256:955bc071aefd3868e6d7292e4f02ee6579f781d66ed2e683354432b0c7de3d8e

WORKDIR /app

RUN python3 -m pip install --no-cache-dir trectools==0.0.50

ADD german-stopwords.txt test_baseline.py baseline.py /app/
COPY query_fields.py test_query_fields.py /app/
COPY candidates.py test_candidates.py /app/
COPY evaluate_submission.py run_spot_checks.py /app/

RUN PYTHONPATH=. pytest .
