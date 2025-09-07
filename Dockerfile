FROM python:3.11.0-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libblas3 \
    libgfortran5 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]