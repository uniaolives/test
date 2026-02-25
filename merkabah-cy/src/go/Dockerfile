FROM golang:1.24-alpine

WORKDIR /app

# Build context is merkabah-cy
COPY src/go/ .

RUN go build -o main .

CMD ["./main"]
