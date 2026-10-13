FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY sydney_tmd_v2/ ./sydney_tmd_v2/
RUN pip install --no-cache-dir .
COPY experiment.json ./
CMD ["python", "-m", "sydney_tmd_v2", "--help"]
