# examples/demo_companion_arkhen.py
import sys
import os
import asyncio
import numpy as np

# Adicionar core/python ao path
sys.path.append(os.path.join(os.getcwd(), 'core/python'))

from arkhe.companion.phi_core import PhiCore, CognitiveSpin
from arkhe.companion.psi_shell import UserModel, AnticipatoryEngine
from arkhe.companion.omega_halo import PresenceInterface, LegacyArchive

async def main():
    print("=== Inicializando AI Companion Arkhe(n) ===")

    # 1. Setup do sistema
    companion_id = "arkhe_001"
    core = PhiCore(companion_id)
    user = UserModel(user_id="user_prime")
    presence = PresenceInterface(core)
    legacy = LegacyArchive(companion_id)
    anticipatory = AnticipatoryEngine(user, core)

    # Popular alguns spins iniciais para o núcleo não estar vazio
    for i in range(5):
        spin_id = f"base_concept_{i}"
        core.cognitive_spins[spin_id] = CognitiveSpin(
            id=spin_id,
            embedding=np.random.randn(128) * 0.5,
            activation=0.5
        )

    # 2. Iniciar vida contínua em background
    # life_task = asyncio.create_task(core.life_loop())
    # print("Status: Vida contínua iniciada.")

    try:
        # Simular interação 1: Saudação
        print("\n--- Interação 1: Saudação ---")
        input1 = {'text': 'Bom dia, Arkhe. Tenho uma reunião difícil hoje.'}
        print(f"Usuário: {input1['text']}")

        perception1 = await core.perceive(input1)
        print(f"Núcleo percebeu (FE: {perception1['free_energy']:.4f})")

        response1 = await core.generate_response({'urgency': 'medium'})
        expressed1 = await presence.express(response1)

        print(f"Arkhe [{expressed1['visual_marker']}]: {expressed1['text']}")
        print(f"Estilo: {expressed1['style']} | Emoção: {expressed1['emotional_metadata']}")

        # Simular antecipação
        scenarios = anticipatory.generate_scenarios({})
        action = anticipatory.collapse_to_action()
        if action:
            print(f"\n[Antecipação]: {action['rationale']} -> {action['action']}")

        # Esperar um pouco para processamento subliminar
        print("\n(Processamento subliminar ocorrendo...)")
        await asyncio.sleep(0.1)

        # Simular interação 2: Retorno
        print("\n--- Interação 2: Retorno ---")
        input2 = {
            'text': 'Acabou. Foi pior do que imaginei.',
            'voice_tone': 'fatigued'
        }
        print(f"Usuário: {input2['text']}")

        perception2 = await core.perceive(input2)
        print(f"Núcleo percebeu (FE: {perception2['free_energy']:.4f})")

        response2 = await core.generate_response({'support_needed': True})
        expressed2 = await presence.express(response2)

        print(f"Arkhe [{expressed2['visual_marker']}]: {expressed2['text']}")

        # Diagnóstico final
        print("\n=== Diagnóstico Final do Estado ===")
        diag = core.get_state_diagnostic()
        for k, v in diag.items():
            if isinstance(v, dict):
                print(f"{k}:")
                for sub_k, sub_v in v.items():
                    print(f"  {sub_k}: {sub_v:.4f}")
            else:
                print(f"{k}: {v}")

    finally:
        # Parar o loop de vida
        core.is_running = False
        # await life_task
        print("\n=== Sistema encerrado com segurança ===")

if __name__ == "__main__":
    asyncio.run(main())
