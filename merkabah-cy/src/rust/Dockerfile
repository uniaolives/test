FROM rust:1.81-slim

WORKDIR /app

# Build context is merkabah-cy
COPY src/rust/ .

RUN cargo build --release

CMD ["./target/release/merkabah-engine"]
