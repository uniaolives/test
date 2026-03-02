# Stage 1: Build Rust components
FROM rust:1.80-slim as builder

WORKDIR /app
COPY . .

# Build all workspace members
RUN cargo build --release -p arkhed -p arkhe-cli -p arkhe-pkg

# Stage 2: Final image
FROM node:20-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    ssh \
    redis-server \
    ipfs \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /etc/arkhe /var/lib/arkhe /mnt/ledger /run/arkhe

# Copy built binaries from builder
COPY --from=builder /app/target/release/arkhed /usr/local/bin/
COPY --from=builder /app/target/release/arkhe-cli /usr/local/bin/
COPY --from=builder /app/target/release/arkhe-pkg /usr/local/bin/

# Create symlinks
RUN ln -s /usr/local/bin/arkhe-cli /usr/local/bin/arkhe-phi && \
    ln -s /usr/local/bin/arkhe-cli /usr/local/bin/arkhe-handover && \
    ln -s /usr/local/bin/arkhe-cli /usr/local/bin/arkhe-test

# Build Companion UI
COPY arkhe_companion/ui /app/ui
WORKDIR /app/ui
RUN npm install && npm run build

# Default configuration (Mocked for now)
RUN echo "peer=127.0.0.1:4242" > /etc/arkhe/peers.conf

# Expose ports
EXPOSE 8080 22 6379 5001

WORKDIR /app

# Entrypoint script
RUN echo '#!/bin/bash\n\
# Start services\n\
redis-server --daemonize yes\n\
# ipfs init && ipfs daemon &\n\
\n\
# Start Arkhe daemon\n\
arkhed &\n\
\n\
# Start Companion UI\n\
cd /app/ui && npm start\n' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
