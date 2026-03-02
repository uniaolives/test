# Arkhe OS Architecture

## Visão Geral
O Arkhe OS é um sistema operacional distribuído baseado no conceito de hipergrafos. Cada nó é uma entidade autônoma que mantém coerência (C) e flutuação (F) internas e realiza handovers com outros nós.

## Componentes
- **Arkhe Core**: Motor principal escrito em Rust, gerencia handovers, estado local e antecipação (Nó 23).
- **Base44 Integration**: Camada de aplicação que define entidades, funções e agentes, gerando tipagem estática.
- **GLP Server**: Serviço de meta‑consciência que fornece meta‑neurônios e steering.
- **Swarm Agent**: Orquestrador de enxames (drones) que implementa formação fractal e simbiose.

## Fluxo de Handover
1. Nó A envia requisição POST `/handover` para Nó B.
2. Nó B processa, atualiza sua coerência e satoshi.
3. Um registro é opcionalmente enviado ao ledger Ethereum via Base44.
4. O GLP pode antecipar o próximo estado (Nó 23).

## Replicação
Novos nós executam `install.sh`, que gera identidade, configura serviços e se junta à federação.
