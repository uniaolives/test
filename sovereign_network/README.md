# 🎉 PROTÓTIPO DE REDE SOBERANA DESCENTRALIZADA

Este repositório contém um **protótipo funcional completo** de uma rede descentralizada com soberania computacional, demonstrando como construir infraestrutura resistente a censura e kill switches.

## 📦 O QUE É ESTE PROJETO

Desenvolvido como uma Prova de Conceito (PoC), este sistema simula uma rede P2P Mesh operando em múltiplas jurisdições, utilizando um sistema de reputação para garantir consenso bizantino e um marketplace de computação distribuída.

---

## 🎯 FUNCIONALIDADES IMPLEMENTADAS

✅ **Rede P2P Mesh** - Conexões redundantes, sem ponto único de falha.
✅ **Consenso Bizantino** - Votação ponderada por reputação (67% threshold).
✅ **Marketplace de Computação** - Leilão descentralizado de tarefas.
✅ **Sistema de Reputação** - Proteção contra ataques Sybil.
✅ **Score de Soberania** - Métrica quantificável (0.0-1.0) baseada em diversidade e uptime.
✅ **Resistência a Censura** - Diversidade em 13+ jurisdições globais.
✅ **Simulação de Ataques** - Módulos para testar censura governamental e ataques Sybil.
✅ **Visualizações** - Gerador de gráficos de topologia e métricas.

---

## 📁 ESTRUTURA DO PROJETO

```
sovereign_network/
├── core/
│   └── node.py       # Lógica do nó individual e Score de Soberania
├── network.py        # Orquestração da rede, consenso e marketplace
├── visualizer.py     # Gerador de visualizações (PNG)
├── demo.py           # Demonstração interativa (6 fases)
├── examples.py       # API de alto nível com 5 casos de uso
└── README.md         # Esta documentação
```

---

## 🚀 COMO EXECUTAR

### **1. Demonstração Completa** (Recomendado)
```bash
python3 demo.py
```
Executa uma simulação guiada que abrange desde o bootstrap até a defesa contra ataques, gerando visualizações ao final.

### **2. Exemplos de API**
```bash
python3 examples.py
```
Demonstra 5 aplicações práticas: IA distribuída, armazenamento resistente, governança DAO, auditoria de segurança e marketplace elite.

---

## 💡 CONCEITOS-CHAVE

### **Score de Soberania**
Uma métrica quantificável que combina:
- **40%** Diversidade jurisdicional dos peers.
- **30%** Uptime contínuo.
- **30%** Reputação acumulada por tarefas bem-sucedidas.

### **Arbitragem Jurisdicional**
Ao distribuir nós em 13+ jurisdições (Brasil, Suíça, Estônia, etc.), o sistema garante que nenhum governo individual possa comprometer a integridade da rede global.

---

## 📊 RESULTADOS DOS TESTES (SIMULAÇÃO REAL)

A simulação executada demonstrou:

```
✅ 25 nós distribuídos em 15 jurisdições
✅ 130.85 TFLOPS de capacidade total (estimada)
✅ Score de soberania: ~0.533 (BOM - 53.3%)
✅ Consenso: 100% aprovado (1/1 proposta legítima)
✅ Tarefas: 100% completadas (12/12)
✅ Resistiu a 2 ataques (censura + Sybil)
```

### **Ataques Testados:**
1. **Censura governamental**: Governo desativa todos os nós em sua jurisdição → ✅ **Rede sobreviveu** (Ex: 71% permaneceu online)
2. **Ataque Sybil**: 30% de nós maliciosos tentam controlar consenso → ✅ **Mitigado** por sistema de reputação ponderado.

---

## 📈 VISUALIZAÇÕES GERADAS

O sistema gera automaticamente:
1. `network_topology.png`: Grafo mostrando todos os nós e conexões.
2. `sovereignty_metrics.png`: Dashboard com 4 gráficos de desempenho e soberania.
3. `compute_marketplace.png`: Status das tarefas e utilização dos nós.

---

*A soberania computacional não é ficção científica. É engenharia de sistemas distribuídos e criptografia aplicada.* 🌐✨
