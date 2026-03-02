# demo.py
"""
Script de Demonstração Interativa
Este script conduz o usuário por uma jornada de 6 fases, simulando a vida útil
de uma rede descentralizada, desde sua criação até sua defesa contra ataques reais.
"""
import time
import random
import sys
import logging
from network import SovereignNetwork
from visualizer import NetworkVisualizer

# Configuração de cores para terminal (ANSI)
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*len(text)}")
    print(text)
    print(f"{'='*len(text)}{Colors.ENDC}")

def run_simulation():
    print(f"{Colors.OKCYAN}{Colors.BOLD}")
    print("      🌐 PROTÓTIPO DE REDE SOBERANA DESCENTRALIZADA 🌐")
    print("      ------------------------------------------------")
    print(f"      Defesa, Soberania e Computação Resistente a Censura{Colors.ENDC}")

    # Fase 1: Bootstrap
    print_header("[FASE 1] BOOTSTRAP DA INFRAESTRUTURA")
    print("Inicializando nós P2P em múltiplas jurisdições...")
    net = SovereignNetwork(25)
    time.sleep(1.5)
    metrics = net.get_network_metrics()
    print(f"{Colors.OKGREEN}✅ Rede online com {metrics['online_nodes']} nós em {metrics['jurisdiction_diversity']} jurisdições.{Colors.ENDC}")
    print(f"✅ Capacidade total agregada: {Colors.BOLD}{metrics['total_capacity_tflops']} TFLOPS{Colors.ENDC}")
    print(f"✅ Top jurisdições: {', '.join([f'{j}({c})' for j, c in metrics['top_jurisdictions']])}")

    # Fase 2: Marketplace de Computação
    print_header("[FASE 2] MARKETPLACE DE COMPUTAÇÃO DISTRIBUÍDA")
    print("Distribuindo tarefas para nós de computação verificados...")
    for i in range(12):
        difficulty = random.uniform(0.1, 0.4)
        net.add_task(difficulty, 10.0)
        if i % 3 == 0: print(f" - Processando lote de tarefas #{i//3 + 1}...")
        time.sleep(0.3)

    completed = [t for t in net.tasks if t["status"] == "completed"]
    print(f"{Colors.OKGREEN}✅ {len(completed)}/12 tarefas completadas com sucesso.{Colors.ENDC}")

    # Fase 3: Consenso Bizantino
    print_header("[FASE 3] CONSENSO BIZANTINO PONDERADO")
    proposal = "Habilitar Sharding para Escalabilidade Orbital (v2.1)"
    print(f"Proposta em votação: {Colors.BOLD}'{proposal}'{Colors.ENDC}")
    print("Coletando votos dos validadores online...")
    time.sleep(1)

    success = net.run_consensus(proposal)
    support = net.proposals[-1]["weight_support"] * 100
    print(f"✅ Resultado: {Colors.BOLD}{'APROVADO' if success else 'REJEITADO'}{Colors.ENDC}")
    print(f"✅ Suporte ponderado por reputação: {support:.1f}% (Threshold: 67%)")

    # Fase 4: Ataque de Censura
    print_header("[FASE 4] SIMULAÇÃO DE ATAQUE: CENSURA GOVERNAMENTAL")
    target = "Brasil"
    print(f"{Colors.WARNING}ALERTA: O governo em '{target}' detectou a rede e emitiu um Kill Switch.{Colors.ENDC}")
    print("Tentando derrubar todos os nós na jurisdição...")
    time.sleep(1.2)

    affected = net.simulate_censorship(target)
    print(f"⚠️  {affected} nós ficaram offline forçadamente.")

    metrics_after = net.get_network_metrics()
    survival_pct = (metrics_after['online_nodes'] / metrics['online_nodes']) * 100
    print(f"{Colors.OKGREEN}✅ Resiliência: {survival_pct:.1f}% da rede permaneceu funcional.{Colors.ENDC}")
    print(f"✅ Nova Soberania Média: {metrics_after['avg_sovereignty']} (Recalculada)")

    # Fase 5: Ataque Sybil
    print_header("[FASE 5] SIMULAÇÃO DE ATAQUE: INJEÇÃO SYBIL")
    print(f"{Colors.WARNING}ALERTA: Atacante tentando injetar 10 nós maliciosos para controlar o consenso.{Colors.ENDC}")
    sybil_ids = net.simulate_sybil_attack(10)
    time.sleep(1)

    print("⚠️  Atacante submete proposta maliciosa: 'Desviar Tesouraria da Rede'")
    malicious_success = net.run_consensus("Transferir fundos para Atacante_Anon")

    if malicious_success:
        print(f"{Colors.FAIL}❌ FALHA CRÍTICA: O ataque Sybil venceu o consenso.{Colors.ENDC}")
    else:
        print(f"{Colors.OKGREEN}✅ ATAQUE MITIGADO: O peso da reputação dos nós legítimos impediu a aprovação.{Colors.ENDC}")
        print("✅ Sistema de Reputação validou a integridade da rede.")

    # Fase 6: Visualização e Métricas Finais
    print_header("[FASE 6] FINALIZAÇÃO E GERAÇÃO DE RELATÓRIOS")
    print("Exportando dashboards de visualização...")
    vis = NetworkVisualizer()
    vis.generate_topology(net, "network_topology.png")
    vis.generate_metrics(net, "sovereignty_metrics.png")
    vis.generate_marketplace(net, "compute_marketplace.png")
    time.sleep(1)

    final_metrics = net.get_network_metrics()
    print(f"\n{Colors.BOLD}📊 RESUMO EXECUTIVO DO PROTÓTIPO{Colors.ENDC}")
    print("-" * 40)
    print(f"Capacidade Online Atual: {Colors.OKBLUE}{final_metrics['total_capacity_tflops']} TFLOPS{Colors.ENDC}")
    print(f"Score de Soberania (φ):  {Colors.OKBLUE}{final_metrics['avg_sovereignty']}{Colors.ENDC}")
    print(f"Jurisdições Ativas:      {final_metrics['jurisdiction_diversity']}")
    print(f"Nós Totais (c/ Sybil):   {final_metrics['total_nodes']}")
    print("-" * 40)
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}✅ PROTÓTIPO ENTREGUE COM SUCESSO!{Colors.ENDC}")
    print(f"{Colors.OKCYAN}Visualize os arquivos PNG gerados para análise detalhada.{Colors.ENDC}\n")

if __name__ == "__main__":
    try:
        run_simulation()
    except KeyboardInterrupt:
        print("\n\nSimulação interrompida pelo usuário.")
        sys.exit(0)
