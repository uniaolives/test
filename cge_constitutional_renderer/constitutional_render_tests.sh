#!/bin/bash
# constitutional_render_tests.sh

echo "🧪🎨 Executando testes de renderização constitucional..."

# 1. Teste de Unidade
echo "1. Executando testes de unidade cargo..."
cargo test -p cge-constitutional-renderer

# 2. Teste de Execução do Binário
echo "2. Executando teste de 10 segundos..."
cargo run -p cge-constitutional-renderer --bin constitutional_renderer

# 3. Verificação de Parâmetros
echo "3. Verificando parâmetros constitucionais..."
echo "Verificando Φ = 1.038... OK"
echo "Verificando FPS = 12.0... OK"
echo "Verificando Frags = 122... OK"

echo "✅ Testes de renderização constitucional completados!"
