# Set a specific Python base image version to ensure compatibility
FROM python:3.11.0-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed for libraries like librosa and pydub
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libblas3 \
    libgfortran5 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all other project files, including your models and app.py
COPY . .

# Expose the port the app will run on
EXPOSE 8000

# Command to run the application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]