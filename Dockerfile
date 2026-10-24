FROM ghcr.io/astral-sh/uv:0.12.23 AS uv

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    HF_HOME=/opt/hf PATH="/app/.venv/bin:$PATH" \
    TINK_DATA=/data PORT=8000 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml uv.lock ./
# --no-cache keeps uv's ~1 GB download cache out of the image
RUN uv sync --frozen --no-dev --no-install-project --no-cache

# Bake the model into the image so containers start without downloading.
RUN python -c "from transformers import AutoFeatureExtractor, AutoModel; m='alkiskoudounas/voc2vec-hubert-ls-pt'; AutoFeatureExtractor.from_pretrained(m); AutoModel.from_pretrained(m)"

COPY tink ./tink
COPY web ./web

RUN useradd -m -u 1000 tink && mkdir /data && chown tink /data && chmod -R a+rX /opt/hf
USER tink
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/api/sounds' % os.environ.get('PORT','8000'), timeout=4)"

# Single worker: the app keeps per-process state.
CMD ["sh", "-c", "exec uvicorn tink.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
