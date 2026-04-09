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

# Change the last line of your Dockerfile to this:
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]