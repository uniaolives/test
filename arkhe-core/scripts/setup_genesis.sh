#!/bin/bash
# setup_genesis.sh

echo "=== ARKHE(N) Genesis Setup ==="

# Dependências
sudo apt update
sudo apt install -y build-essential cmake libsqlite3-dev nlohmann-json3-dev

# Biblioteca httplib (header-only)
wget https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h  -O /tmp/httplib.h
sudo cp /tmp/httplib.h /usr/local/include/

# Compilação
g++ -std=c++17 -O3 -o arkhe_genesis arkhe_genesis.cpp \
    -lsqlite3 -lpthread -lssl -lcrypto

echo "=== Compilação concluída ==="
echo "Execute: ./arkhe_genesis --server 8080"
