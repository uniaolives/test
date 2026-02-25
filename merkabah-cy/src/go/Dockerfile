FROM golang:1.21-alpine

WORKDIR /app

# Build context is merkabah-cy
COPY src/go/ .

RUN go build -o main .

CMD ["./main"]
