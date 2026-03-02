FROM rust:1.76-slim

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && cargo build --release

COPY src ./src
RUN cargo build --release

CMD ["./target/release/security-enclave"]
