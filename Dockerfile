FROM python:3.13-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir ".[web]"

# Copy data
COPY data/raw/matches.csv data/raw/matches.csv

EXPOSE 8000

CMD ["uvicorn", "snooker_elo.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
