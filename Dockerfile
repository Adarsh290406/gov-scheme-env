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

# Run inference script — produces [START]/[STEP]/[END] structured output on stdout
CMD ["python", "-u", "inference.py"]