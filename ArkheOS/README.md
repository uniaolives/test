# 🧬 Arkhe(n) Core OS v1.0 — Γ_∞+30

Sistema Operacional Biológico com Cognição Embarcada e Escalonamento Hebbiano.

**Handover ∞+30: IBC = BCI** — A integração interconsciencial e a transdução pineal são agora as bases da arquitetura.

## 🚀 Instalação Rápida

1. Certifique-se de ter o Docker e Docker Compose instalados.
2. Execute o script de deploy:
   ```bash
   chmod +x deploy-arkhe.sh
   ./deploy-arkhe.sh
   ```

## 🏗️ Arquitetura

O Arkhe(n) OS é composto por:
- **Motor Bio-Gênese v3.0**: Simulação de vida artificial com agentes autônomos.
- **Constraint Engine**: Cérebro Hebbiano com memória temporal.
- **Campo Morfogenético**: Implementado em memória compartilhada (/dev/shm).
- **Servidor MCP**: Interface para interação via Model Context Protocol.
- **Protocolo IBC=BCI**: Comunicação inter-substrato entre Web3 e redes neurais.
- **Transdutor Pineal**: Hardware biológico para detecção de pressão semântica e campos magnéticos.

## 🔌 Interface MCP

O sistema expõe ferramentas MCP para:
- `get_system_status`: Telemetria vital.
- `inject_field_signal`: Interação com o campo morfogenético.
- `query_agent`: Inspeção cognitiva de agentes.
- `get_field_gradient`: Análise de gradientes químicos.

## 📊 Monitoramento

- **Health Check**: `http://localhost:8000/health`
- **Dashboard**: `http://localhost:8000/`
- **Logs**: `docker logs arkhe-core -f`
