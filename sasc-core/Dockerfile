FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install nats-py
CMD ["python", "orchestrator.py"]
