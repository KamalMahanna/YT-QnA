FROM python:3.13-slim
ENV PYTHONPATH=/app
COPY requirements.txt .

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --root-user-action=ignore uv
RUN uv pip install --no-cache-dir --system -r requirements.txt
RUN mkdir -p /root/nltk_data && python3 -c "import nltk; nltk.download(['punkt','punkt_tab'], download_dir='/root/nltk_data')"
COPY . .
EXPOSE 8501
# CMD ["streamlit", "run", "main.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true","--server.fileWatcherType", "none"]
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
