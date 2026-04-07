FROM python:3.10-slim-bookworm

WORKDIR /app

# Force stdout to be unbuffered so [START]/[STEP]/[END] lines are captured immediately
ENV PYTHONUNBUFFERED=1

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 5 --timeout 60 -r requirements.txt

# Copy application files
COPY . .

EXPOSE 7860

# Run inference script — produces [START]/[STEP]/[END] structured output on stdout
CMD ["python", "-u", "inference.py"]
