# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

FROM base AS runtime
COPY infra/docker/requirements-lock.txt ./
RUN pip install --break-system-packages --no-cache-dir --upgrade pip==24.0 && \
    pip install --break-system-packages --no-cache-dir --requirement requirements-lock.txt

COPY pyproject.toml ./
COPY apps ./apps
COPY shared ./shared
COPY ml ./ml
RUN pip install --break-system-packages --no-cache-dir --no-deps . && \
    useradd --create-home --uid 1000 appuser

USER 1000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
