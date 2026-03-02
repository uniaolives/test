FROM python:3.10-slim
RUN apt-get update && apt-get install -y libpcap-dev gcc && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY security_agent.py .
RUN pip install nats-py scapy psutil
CMD ["python", "security_agent.py"]
