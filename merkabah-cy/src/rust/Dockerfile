FROM rust:1.70-slim

WORKDIR /app

COPY . .

RUN cargo build --release

CMD ["./target/release/merkabah-engine"]
