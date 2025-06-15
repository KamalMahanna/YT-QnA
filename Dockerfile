FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN apt update && apt install -y ffmpeg
RUN pip install --root-user-action=ignore uv
RUN uv pip install --no-cache-dir --system -r requirements.txt

COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "main.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]
