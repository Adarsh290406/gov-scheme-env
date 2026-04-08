FROM python:3.10-slim-bookworm

WORKDIR /app

# Unbuffered stdout (important for validator)
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 5 --timeout 60 -r requirements.txt

# Copy all files
COPY . .

# Expose port (HF expects it, even if unused)
EXPOSE 7860

# Run inference AND keep container alive so logs are captured
CMD ["sh", "-c", "python -u inference.py && sleep 60"]