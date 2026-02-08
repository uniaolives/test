FROM ubuntu:22.04
RUN apt-get update && apt-get install -y g++ make git
WORKDIR /app
COPY . .
RUN cd src/ancient_scripts && make clean && make
CMD ["./bin/phoenician_simulator"]
