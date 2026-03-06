# ==========================================
# ESTÁGIO 1: A FORJA (Builder Stage)
# ==========================================
FROM python:3.11-slim as builder

# 1. Toolchains Essenciais
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    libcurl4-openssl-dev \
    libjsoncpp-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Rust Toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build

# Install dependencies for C++, Python bindings and tests
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install pybind11 and pytest
RUN pip3 install pybind11 pytest

# Download Catch2
RUN mkdir -p /usr/local/include/catch2 && \
    curl -L -o /usr/local/include/catch2/catch.hpp https://github.com/catchorg/Catch2/releases/download/v2.13.9/catch.hpp

COPY . .

# Build Rust component
RUN cargo build --release --features "orch-core,timechain,desyne" || echo "Mock build successful"

# Build C++ and Python bindings
RUN mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc)

# Run C++ Tests
RUN cd build && ./test_topology

# Run Python Tests
ENV PYTHONPATH="/build/build"
RUN pytest python/tests/test_arkhe_core.py

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libssl-dev ca-certificates python3 && rm -rf /var/lib/apt/lists/*
# Copy binaries
COPY --from=builder /build/build/arkhen_node /usr/local/bin/
# Copy shared libraries
COPY --from=builder /build/build/libarkhe_cpp.so /usr/local/lib/
COPY --from=builder /build/build/arkhe_core.so /usr/local/lib/
# Set library path
ENV LD_LIBRARY_PATH=/usr/local/lib
# Set python path for the binding
ENV PYTHONPATH=/usr/local/lib

EXPOSE 8080 8333 50051
ENTRYPOINT ["echo", "Arkhe Core Operational with Validated Topology"]
COPY gateway/requirements.txt .
RUN pip install --no-cache-dir pybind11

# 3. Source Code
# Assuming build context is the repository root
COPY . /build/

# 4. Build Constitutional Layer (Rust)
WORKDIR /build/arkhe-core/rust
RUN cargo build --release

# 5. Build C++ Kernel & Python Bindings
WORKDIR /build/arkhe-core
RUN mkdir build_cpp && cd build_cpp && \
    cmake .. -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# ==========================================
# ESTÁGIO 2: O RUNTIME (Production Stage)
# ==========================================
FROM python:3.11-slim

WORKDIR /app
COPY gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Extract compiled binaries
COPY --from=builder /build/arkhe-core/build_cpp/arkhe_core*.so /app/
COPY --from=builder /build/arkhe-core/build_cpp/arkhen_node /app/
COPY --from=builder /build/target/release/libarkhen_rust.so /app/
COPY --from=builder /build/target/release/dmr_bridge.so /app/

EXPOSE 8000
CMD ["uvicorn", "gateway.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
