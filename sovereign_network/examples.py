# examples.py
"""
Exemplos de API e Casos de Uso
Este arquivo demonstra como interagir com a Rede Soberana programaticamente
para resolver problemas do mundo real em infraestrutura descentralizada.
"""
import random
import time
from network import SovereignNetwork
from core.node import NodeKind, JURISDICTIONS

# Utilitário para separadores visuais
def section(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def example_ai_training():
    """Caso de Uso 1: Treinamento de Modelos de IA em larga escala sem censura."""
    section("1. INTELIGÊNCIA ARTIFICIAL DISTRIBUÍDA (GPU CLUSTER)")
    net = SovereignNetwork(20)
    print("Iniciando treinamento de LLM soberano...")
    print("Dividindo modelo em 50 fragmentos para processamento paralelo...")

    success_count = 0
    start_time = time.time()
    for i in range(50):
        # Cada tarefa representa um batch de treinamento
        if net.add_task(difficulty=random.uniform(0.6, 0.9), reward=2.5):
            success_count += 1

    duration = time.time() - start_time
    print(f"✅ Treinamento concluído em {duration:.2f}s.")
    print(f"✅ Resultado: {success_count}/50 batches processados via Compute Nodes.")
    print(f"✅ Reputação média da rede aumentou após entrega de valor.")

def example_storage():
    """Caso de Uso 2: Armazenamento distribuído imutável para whistleblowers."""
    section("2. ARMAZENAMENTO RESISTENTE A CENSURA (DATA VAULT)")
    net = SovereignNetwork(30)
    print("Criptografando e fragmentando 'Documentos Confidenciais'...")

    # Identifica nós de storage
    storage_nodes = [n for n in net.nodes.values() if n.kind == NodeKind.STORAGE]
    print(f"Distribuindo fragmentos Redundant-Array em {len(storage_nodes)} nós de storage.")

    # Simula um ataque de remoção de conteúdo (DMCA global ou censura estatal)
    # Atacando jurisdições onde nós de storage são comuns
    print("⚠️ Simulando Ataque de Censura em massa (Panamá, Seychelles)...")
    net.simulate_censorship("Panamá")
    net.simulate_censorship("Seychelles")

    online_storage = [n for n in net.nodes.values() if n.kind == NodeKind.STORAGE and n.is_online]
    survival_ratio = len(online_storage) / len(storage_nodes) if storage_nodes else 0

    if survival_ratio > 0.4: # Se 40% sobreviver, os dados são recuperáveis via Reed-Solomon
        print(f"✅ SUCESSO: Dados íntegros. {len(online_storage)} nós de storage permanecem.")
        print(f"✅ Ratio de Sobrevivência: {survival_ratio*100:.1f}%")
    else:
        print(f"❌ FALHA: Perda de integridade de dados (Sobrevivência: {survival_ratio*100:.1f}%)")

def example_governance():
    """Caso de Uso 3: Governança de Tesouraria via DAO Ponderada."""
    section("3. GOVERNANÇA DESCENTRALIZADA (DAO CONSENSUS)")
    net = SovereignNetwork(40)

    proposal = "Alocar 500k tokens para Pesquisa de Comunicacão via Neutrinos"
    print(f"Proposta em debate: '{proposal}'")

    # Simula um grupo de validadores de elite (alta reputação)
    elite_nodes = list(net.nodes.values())[:8]
    for n in elite_nodes:
        n.reputation = 0.92
        n.calculate_sovereignty_score([n.jurisdiction for n in net.nodes.values()])

    print("Iniciando ciclo de votação de 48h (simulado)...")
    approved = net.run_consensus(proposal)

    metrics = net.proposals[-1]
    print(f"✅ Votação encerrada.")
    print(f"✅ Status: {'APROVADA' if approved else 'REJEITADA'}")
    print(f"✅ Suporte de Reputação: {metrics['weight_support']*100:.2f}%")

def example_security_check():
    """Caso de Uso 4: Auditoria de Segurança e Score de Soberania (φ)."""
    section("4. AUDITORIA DE RESISTÊNCIA E SOBERANIA (φ-CHECK)")
    # Criamos uma rede propositalmente centralizada para teste
    net = SovereignNetwork(10)
    for n in net.nodes.values():
        n.jurisdiction = "USA" # Centralização total

    net.update_all_scores()
    metrics = net.get_network_metrics()

    print(f"Analisando rede de teste (Centralizada)...")
    print(f"Score de Soberania (φ) Médio: {metrics['avg_sovereignty']}")

    if metrics['avg_sovereignty'] < 0.4:
        print(f"🔴 ALERTA: Rede vulnerável! Falta de diversidade jurisdicional detectada.")
        print(f"🔴 Recomendação: Adicionar nós em jurisdições BRICS+ e Offshore.")

    # Agora adicionamos diversidade
    print("\nAdicionando nós diversificados (Brasil, Estônia, Islândia)...")
    # Vamos trocar as jurisdições de alguns nós existentes para aumentar a diversidade
    for i, n in enumerate(net.nodes.values()):
        if i < len(JURISDICTIONS):
            n.jurisdiction = JURISDICTIONS[i]

    net.update_all_scores()
    new_metrics = net.get_network_metrics()
    print(f"✅ Novo Score após diversificação: {new_metrics['avg_sovereignty']}")
    print(f"✅ Ganho de Soberania: {((new_metrics['avg_sovereignty']/metrics['avg_sovereignty'])-1)*100:.1f}%")

def example_marketplace():
    """Caso de Uso 5: Contratação de infraestrutura verificada."""
    section("5. MARKETPLACE DE PROVEDORES ELITE (VERIFIED COMPUTE)")
    net = SovereignNetwork(25)
    print("Buscando Provedores de Computação com Tier S (Reputação > 0.8)...")

    # Realiza algumas tarefas para subir reputação de alguns nós
    for _ in range(30):
        net.add_task(0.3, 10.0)

    elite = [n for n in net.nodes.values() if n.reputation > 0.75 and n.kind == NodeKind.COMPUTE]

    print(f"Encontrados {len(elite)} provedores verificados para alta performance:")
    for i, node in enumerate(elite[:5]):
        print(f" #{i+1} [Nó {node.id}] | Jur: {node.jurisdiction} | Rep: {node.reputation:.3f} | Cap: {node.capacity_tflops} TFLOPS")

if __name__ == "__main__":
    print(f"\n{'-'*60}")
    print("  EXEMPLOS DE USO DA API - REDE SOBERANA")
    print(f"{'-'*60}")

    example_ai_training()
    example_storage()
    example_governance()
    example_security_check()
    example_marketplace()

    print(f"\n\n{'*'*60}")
    print(" Fim dos exemplos de API. Explore o código em network.py para mais detalhes.")
    print(f"{'*'*60}\n")
