FROM python:3.13-slim
ENV PYTHONPATH=/app
COPY requirements.txt .

RUN apt update && apt install -y ffmpeg
RUN pip install --root-user-action=ignore uv
RUN uv pip install --no-cache-dir --system -r requirements.txt
COPY . .
CMD ["streamlit", "run", "main.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]
