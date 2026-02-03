#!/bin/bash
# Monitoramento SafeCore-9D

INTERVAL=${1:-5}  # Segundos entre verificações

echo "🔍 Monitor SafeCore-9D iniciado (intervalo: ${INTERVAL}s)"
echo ""

while true; do
    clear

    # Cabeçalho
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                SAFECORE-9D MONITOR v1.0                     ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    # 1. Verificar processo
    if pgrep -f "safecore-9d" > /dev/null; then
        PID=$(pgrep -f "safecore-9d" | head -1)
        echo "✅ Processo ativo (PID: $PID)"

        # Uso de recursos
        if command -v ps > /dev/null; then
            CPU=$(ps -p $PID -o %cpu --no-headers 2>/dev/null || echo "N/A")
            MEM=$(ps -p $PID -o %mem --no-headers 2>/dev/null || echo "N/A")
            echo "   CPU: ${CPU}% | Memória: ${MEM}%"
        fi
    else
        echo "❌ Processo não encontrado"
    fi

    echo ""

    # 2. Verificar portas
    echo "🌐 Portas de Serviço:"

    check_port_with_name() {
        local port=$1
        local name=$2

        if command -v ss > /dev/null 2>&1; then
            if ss -tuln 2>/dev/null | grep -q ":$port "; then
                echo "   ✅ $name (porta $port): ATIVA"
            else
                echo "   ❌ $name (porta $port): INATIVA"
            fi
        else
            echo "   ⚠️  $name (porta $port): DESCONHECIDO (ss not found)"
        fi
    }

    # Portas padrão
    check_port_with_name 9050 "Dashboard"
    check_port_with_name 9100 "Métricas"
    check_port_with_name 9150 "Ética"

    echo ""

    # 3. Verificar recursos do sistema
    if command -v free > /dev/null; then
        echo "💾 Uso de Memória:"
        free -h | awk 'NR==2{printf "   Total: %s | Usado: %s | Livre: %s\n", $2, $3, $4}'
    fi

    if command -v df > /dev/null; then
        echo ""
        echo "💿 Espaço em Disco:"
        df -h / | awk 'NR==2{printf "   Uso: %s de %s (%s)\n", $3, $2, $5}'
    fi

    echo ""

    # 4. Timestamp
    echo "🕐 Última verificação: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "   Próxima em: ${INTERVAL} segundos"

    sleep $INTERVAL
done
