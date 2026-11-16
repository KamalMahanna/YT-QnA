FROM python:3.13-slim
ENV PYTHONPATH=/app
ENV PORT=8000

COPY requirements.txt .

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --root-user-action=ignore uv
RUN uv pip install --no-cache-dir --system -r requirements.txt
RUN mkdir -p /root/nltk_data && python3 -c "import nltk; nltk.download(['punkt','punkt_tab'], download_dir='/root/nltk_data')"

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
