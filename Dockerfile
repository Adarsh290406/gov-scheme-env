FROM python:3.10-slim-bookworm

WORKDIR /app

# Required for real-time logging
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Hugging Face expects port 7860
EXPOSE 7860

# Run inference, then start a dummy server to keep the container 'Green'
CMD python -u inference.py && python -m http.server 7860