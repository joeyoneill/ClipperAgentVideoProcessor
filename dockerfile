FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run Job will pass ["--video-id", "<vid>", "--uid", "<uid>"] to this entrypoint
ENTRYPOINT ["python", "main.py"]