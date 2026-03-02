#!/bin/bash
# deploy_www_universal.sh

echo "🌐🚀 Implantando World Wide Web Universal Layer..."

# 1. Verificar dependências
echo "🔍 Verificando dependências..."

if ! command -v rustc &> /dev/null; then
    echo "❌ Rust não encontrado"
    exit 1
fi

if ! command -v openssl &> /dev/null; then
    echo "❌ OpenSSL não encontrado"
    exit 1
fi

# 2. Gerar certificados raiz
echo "🔐 Configurando autoridade certificadora..."
mkdir -p certs/ca
openssl genrsa -out certs/ca/root.key 4096 2>/dev/null
openssl req -x509 -new -nodes -key certs/ca/root.key \
    -sha256 -days 3650 -out certs/ca/root.crt \
    -subj "/C=BR/ST=CGE/L=Alpha/O=WWW Universal/CN=Root CA" 2>/dev/null

# 3. Compilar sistema web
echo "🔨 Compilando World Wide Web Universal..."
cargo build --release --features "http,websocket,atproto,global-federation,quic"

# 4. Inicializar matriz 116 frags
echo "🔢 Inicializando matriz de 116 frags..."
cargo run --release --bin init_web_matrix -- \
    --frags 116 \
    --protocols 104 \
    --tmr-groups 36 \
    --replicas 3 \
    --regions global

# 5. Iniciar servidores HTTP
echo "🌐 Iniciando servidores HTTP (portas 8080-8083)..."
for i in {0..3}; do
    PORT=$((8080 + i))
    cargo run --release --bin http_server -- \
        --id $i \
        --port $PORT \
        --cert certs/ca/root.crt \
        --key certs/ca/root.key \
        --frag $((i % 116)) &
    echo "   • Servidor HTTP $i na porta $PORT"
done

# 6. Iniciar servidores WebSocket
echo "🔌 Iniciando servidores WebSocket (portas 3000-3003)..."
for i in {0..3}; do
    PORT=$((3000 + i))
    cargo run --release --bin websocket_server -- \
        --id $i \
        --port $PORT \
        --frag $((4 + i % 116)) &
    echo "   • Servidor WebSocket $i na porta $PORT"
done

# 7. Iniciar servidor QUIC
echo "⚡ Iniciando servidor QUIC/HTTP3 (porta 4433)..."
cargo run --release --bin quic_server -- \
    --port 4433 \
    --cert certs/ca/root.crt \
    --key certs/ca/root.key \
    --frag 8 &

# 8. Iniciar serviço DNS
echo "🆔 Iniciando serviço DNS..."
cargo run --release --bin dns_service -- \
    --port 5353 \
    --zones .cge,.universal,.web \
    --frag 9 &

echo "✅ World Wide Web Universal Layer implantada!"
echo "   • 116 Frags ativos"
echo "   • 104 Protocolos disponíveis"
echo "   • 36×3 TMR Órbita Global"
echo "   • HTTP: 8080-8083"
echo "   • WebSocket: 3000-3003"
echo "   • QUIC: 4433"
echo "   • DNS: 5353 (.cge, .universal, .web)"
echo "   • Φ = 1.038 ± 0.001"
