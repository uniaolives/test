# examples/demo_companion_arkhen_refined.py
import sys
import os
import asyncio
import numpy as np
from datetime import datetime

# Adicionar core/python ao path
sys.path.append(os.path.join(os.getcwd(), 'core/python'))

from arkhe.companion import (
    PhiCore, CognitiveSpin, UserModel,
    PresenceInterface, AnticipatoryEngine,
    ConsolidationEngine, NonLinearPersonalityKnob, ContextualPhi,
    DeviceSyncEngine
)

async def main():
    print("=== Inicializando AI Companion Arkhe(n) Refinado ===")

    # 1. Setup do sistema
    companion_id = "arkhe_001"
    core = PhiCore(companion_id)
    user = UserModel(user_id="user_prime")
    presence = PresenceInterface(core)
    anticipatory = AnticipatoryEngine(user, core)

    # Novos motores refinados
    sync = DeviceSyncEngine(device_id="mobile_01")
    knob = NonLinearPersonalityKnob(core)
    context_engine = ContextualPhi(knob)
    consolidation = ConsolidationEngine(core, core.memory)

    # 2. Iniciar vida contínua com Consolidação
    life_task = asyncio.create_task(core.life_loop(consolidation_engine=consolidation))
    print("Status: Vida contínua (com Sonho) iniciada.")

    try:
        # Mock de alguns spins para NMF
        print("\n--- Carregando Base de Conhecimento Inicial ---")
        for i in range(10):
            sid = f"concept_{i}"
            core.cognitive_spins[sid] = CognitiveSpin(
                id=sid,
                embedding=np.random.randn(128),
                activation=0.5
            )

        # Simular interação 1: Trabalho (Foco Analítico)
        print("\n--- Cenário 1: Trabalho (9h00) ---")
        context = context_engine.update_context({'stress_level': 0.2})
        print(f"Contexto Detectado: {context} | Phi: {knob.phi:.4f}")

        input1 = {'text': 'Arkhe, analise o relatório de vendas.'}
        perception1 = await core.perceive(input1)
        response1 = await core.generate_response({})
        expressed1 = await presence.express(response1)
        print(f"Arkhe [{expressed1['visual_marker']}]: {expressed1['text']} (Estilo: {expressed1['style']})")

        # Simular transição para REFLECTIVE (Sonho)
        print("\n--- Transição: Usuário Ocioso (Iniciando Sonho) ---")
        core.operating_state = 'REFLECTIVE'
        print("Aguardando ciclos de consolidação...")
        await asyncio.sleep(1.5) # Deixa o loop de vida rodar alguns ciclos de 100

        # Simular interação 2: Crise (Phi Reduzido para Previsibilidade)
        print("\n--- Cenário 2: Crise detectada ---")
        core.operating_state = 'ACTIVE'
        context2 = context_engine.update_context({'stress_level': 0.9})
        print(f"Contexto Detectado: {context2} | Phi: {knob.phi:.4f}")

        input2 = {'text': 'Estou muito estressado com o prazo.'}
        perception2 = await core.perceive(input2)
        response2 = await core.generate_response({'support': True})
        expressed2 = await presence.express(response2)
        print(f"Arkhe [{expressed2['visual_marker']}]: {expressed2['text']}")

        # Sincronização (Mock)
        print("\n--- Sincronização de Dispositivo ---")
        sync.local_update('personality_phi', knob.phi)
        print(f"Phi sincronizado no LWW-Register: {sync.stores['personality_phi'].value:.4f}")

        # Diagnóstico final
        print("\n=== Diagnóstico Final do Estado ===")
        diag = core.get_state_diagnostic()
        print(f"Phi Operacional: {diag['operational_phi']:.4f}")
        print(f"Spins Ativos: {diag['num_cognitive_spins']}")
        print(f"Entropia de Crença: {diag['belief_entropy']:.4f}")

    finally:
        core.is_running = False
        await life_task
        print("\n=== Sistema encerrado com segurança ===")

if __name__ == "__main__":
    asyncio.run(main())
