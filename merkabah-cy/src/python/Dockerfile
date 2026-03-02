FROM python:3.11-slim

WORKDIR /app

# Build context is merkabah-cy
COPY src/python/ .

RUN pip install --no-cache-dir -e .

CMD ["python", "-m", "merkabah.quantum.qhttp"]
