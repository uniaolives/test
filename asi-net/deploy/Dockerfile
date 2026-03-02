# Dockerfile para nó ASI
FROM rust:1.75-slim AS builder

WORKDIR /usr/src/asi-net
COPY . .

# Compilar componentes Rust
RUN cd asi-net/rust/asi-net-crate && cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libssl-dev ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/src/asi-net/asi-net/rust/asi-net-crate/target/release/asi-net-server /usr/local/bin/asi-net-server

# Expor portas
EXPOSE 2718  # ASI Protocol
EXPOSE 2222  # ASI SSH
EXPOSE 8080  # Webhooks
EXPOSE 9999  # Morphic Resonance

# Iniciar nó
CMD ["asi-net-server"]
