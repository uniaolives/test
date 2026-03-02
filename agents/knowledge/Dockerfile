FROM python:3.10-slim
WORKDIR /app
COPY knowledge_agent.py .
RUN pip install nats-py qdrant-client sentence-transformers numpy
CMD ["python", "knowledge_agent.py"]
