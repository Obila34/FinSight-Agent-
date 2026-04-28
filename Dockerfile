FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PORT=7860

# Basic build tools needed by some scientific Python packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . /app

# Spaces expects the app to listen on 7860.
EXPOSE 7860

# Run Streamlit UI in the Space.
# Set FINSIGHT_API_URL in Space Variables/Secrets to your backend API URL.
CMD ["python", "-m", "streamlit", "run", "ui/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]