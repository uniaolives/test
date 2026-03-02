#!/bin/bash
# Menu Interativo SafeCore-9D

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

show_header() {
    clear
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                   SAFECORE-9D CONTROL PANEL                  ║"
    echo "║                        v9.0.0                                ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

show_menu() {
    echo -e "${CYAN}Selecione uma opção:${NC}"
    echo ""
    echo -e "  ${GREEN}1${NC}) 🚀 Iniciar SafeCore-9D (Modo Desenvolvimento)"
    echo -e "  ${GREEN}2${NC}) 🏭 Iniciar SafeCore-9D (Modo Produção)"
    echo -e "  ${GREEN}3${NC}) 📊 Monitorar Sistema"
    echo -e "  ${GREEN}4${NC}) 🔧 Compilar Projeto"
    echo -e "  ${GREEN}5${NC}) 📝 Ver Logs"
    echo -e "  ${GREEN}6${NC}) ⚖️  Verificar Constituição"
    echo -e "  ${GREEN}7${NC}) 🧪 Executar Testes"
    echo -e "  ${GREEN}8${NC}) 🗑️  Limpar Ambiente"
    echo -e "  ${GREEN}9${NC}) 📋 Status do Sistema"
    echo -e "  ${GREEN}0${NC}) ❌ Sair"
    echo ""
    echo -n -e "${YELLOW}Digite sua escolha: ${NC}"
}

option1() {
    echo -e "\n${GREEN}Iniciando SafeCore-9D em modo desenvolvimento...${NC}"
    ./deploy_9d.sh --dev
}

option2() {
    echo -e "\n${GREEN}Iniciando SafeCore-9D em modo produção...${NC}"
    ./deploy_9d.sh --prod
}

option3() {
    echo -e "\n${GREEN}Iniciando monitoramento...${NC}"
    ./monitor_9d.sh
}

option4() {
    echo -e "\n${GREEN}Compilando projeto...${NC}"
    cargo build --release
    echo -e "${CYAN}Compilação completa!${NC}"
    read -p "Pressione Enter para continuar..."
}

option5() {
    echo -e "\n${GREEN}Últimos logs:${NC}"
    if [ -f "$HOME/.safecore/logs/daemon.log" ]; then
        tail -20 "$HOME/.safecore/logs/daemon.log"
    else
        echo -e "${YELLOW}Arquivo de log não encontrado em $HOME/.safecore/logs/daemon.log${NC}"
    fi
    echo ""
    read -p "Pressione Enter para continuar..."
}

option6() {
    echo -e "\n${GREEN}Verificando constituição...${NC}"
    if [ -f "constitution/constitution.json" ]; then
        python3 -m json.tool constitution/constitution.json 2>/dev/null || \
        cat constitution/constitution.json
        echo ""
        echo -e "${CYAN}Constituição válida!${NC}"
    else
        echo -e "${RED}Constituição não encontrada!${NC}"
    fi
    read -p "Pressione Enter para continuar..."
}

option7() {
    echo -e "\n${GREEN}Executando testes...${NC}"
    cargo test -- --nocapture
    read -p "Pressione Enter para continuar..."
}

option8() {
    echo -e "\n${YELLOW}Limpando ambiente...${NC}"
    cargo clean
    rm -rf target/ 2>/dev/null || true
    echo -e "${GREEN}Ambiente limpo!${NC}"
    read -p "Pressione Enter para continuar..."
}

option9() {
    echo -e "\n${CYAN}Status do Sistema SafeCore-9D:${NC}"
    echo ""

    # Verificar processos
    if pgrep -f "safecore-9d" > /dev/null; then
        echo -e "  ${GREEN}✅ SafeCore-9D: ATIVO${NC}"
        echo -e "     PID: $(pgrep -f 'safecore-9d')"
    else
        echo -e "  ${RED}❌ SafeCore-9D: INATIVO${NC}"
    fi

    # Verificar portas
    echo ""
    echo -e "  ${CYAN}Portas:${NC}"

    for port in 9050 9100 9150; do
        if ss -tuln 2>/dev/null | grep -q ":$port "; then
            echo -e "    ${GREEN}✅ Porta $port: ABERTA${NC}"
        else
            echo -e "    ${YELLOW}⚠️  Porta $port: FECHADA${NC}"
        fi
    done

    # Verificar constituição
    echo ""
    if [ -f "constitution/constitution.json" ]; then
        echo -e "  ${GREEN}✅ Constituição: PRESENTE${NC}"
    else
        echo -e "  ${RED}❌ Constituição: AUSENTE${NC}"
    fi

    # Verificar binário
    echo ""
    if [ -f "target/release/safecore-9d" ]; then
        echo -e "  ${GREEN}✅ Binário: COMPILADO${NC}"
    elif [ -f "target/debug/safecore-9d" ]; then
        echo -e "  ${YELLOW}⚠️  Binário: MODO DEBUG${NC}"
    else
        echo -e "  ${RED}❌ Binário: NÃO COMPILADO${NC}"
    fi

    echo ""
    read -p "Pressione Enter para continuar..."
}

# Menu principal
while true; do
    show_header
    show_menu

    read choice

    case $choice in
        1) option1 ;;
        2) option2 ;;
        3) option3 ;;
        4) option4 ;;
        5) option5 ;;
        6) option6 ;;
        7) option7 ;;
        8) option8 ;;
        9) option9 ;;
        0)
            echo -e "\n${CYAN}Saindo... Adeus! 👋${NC}"
            exit 0
            ;;
        *)
            echo -e "\n${RED}Opção inválida!${NC}"
            sleep 1
            ;;
    esac
done
