FROM rust:1.75-slim-bullseye as builder

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    build-essential \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Copiar arquivos do projeto
COPY ont/native/Cargo.toml ont/native/Cargo.lock* ./
COPY ont/native/src ./src

# Build do projeto com features onchain
RUN cargo build --release --features "onchain"

# Imagem final minimalista
FROM debian:bullseye-slim

WORKDIR /app

# Instalar runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar binário
COPY --from=builder /app/target/release/ontology-lang /usr/local/bin/onto

# Criar usuário não-root
RUN useradd -m -u 1000 ontology && chown -R ontology:ontology /app
USER ontology

# Expor porta para métricas
EXPOSE 8080

# Comando padrão
CMD ["onto", "server", "--metrics-port", "8080"]
