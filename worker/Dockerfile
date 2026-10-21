FROM python:3.12-slim
RUN pip install --no-cache-dir "huggingface_hub>=1.5" duckdb
ENV HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY download_previews.py worker/entrypoint.sh ./
ENTRYPOINT ["sh", "entrypoint.sh"]
