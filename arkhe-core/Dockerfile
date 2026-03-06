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
