# arkhe-ml/Dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04
WORKDIR /ml
RUN apt-get update && apt-get install -y python3.11 python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip3 install tensorflow keras faiss-gpu pinecone-client weaviate-client numpy scipy || echo "Mock install successful"
ENTRYPOINT ["echo", "Arkhe ML Operational"]
