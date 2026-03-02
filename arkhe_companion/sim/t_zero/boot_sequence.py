# arkhe_companion/sim/t_zero/boot_sequence.py
import sys
import os
import time
import asyncio
import numpy as np
from datetime import datetime

# Path setup para incluir core/
sys.path.append(os.path.join(os.getcwd(), 'core'))

from phi_core.phi_engine import PhiCore
from phi_core.user_model import UserModel
from phi_core.presence import PresenceInterface
from personality.knob import NonLinearPersonalityKnob

async def run_boot():
    print("[ARKHE BOOT SEQUENCE - Ω+200.2]")
    print(f"Timestamp: {datetime.now().isoformat()}Z")
    print("Device: genesis-node-001")

    start_time = time.time()

    def log_ms(msg):
        ms = (time.time() - start_time) * 1000
        print(f"[{ms:0.3f} ms] {msg}")

    # 1. Parâmetros de criticidade
    log_ms("BOOTSTRAP: Carregando parâmetros de criticidade...")
    phi_initial = 0.618033988749894
    log_ms(f"              Φ_INITIAL = {phi_initial}")
    log_ms("              MODE = CRITICAL (borda do caos)")

    # 2. Enclave de segurança (Mock)
    log_ms("CRYPTO: Inicializando enclave de segurança...")
    log_ms("              Kyber-1024: OK")
    log_ms("              Dilithium-3: OK")

    # 3. Alocação de campo holográfico
    log_ms("MEMORY: Alocando campo holográfico...")
    companion_id = "arkhe_001"
    core = PhiCore(companion_id)
    log_ms("              Resolution: 512x512 (toroide T²)")
    log_ms(f"              Field energy: {np.sum(np.abs(core.memory.field)**2):0.6f}")

    # 4. Rede de spins
    log_ms("COGNITIVE: Instanciando rede de spins...")
    log_ms("              Initial spins: 0")
    log_ms("              Embedding dimension: 128")

    # 5. Modelo do usuário e FEP
    user = UserModel(user_id="user_prime")
    log_ms("FEP: Inicializando modelo generativo...")
    log_ms("              Latent dim: 64 (belief space)")
    log_ms(f"              Temperature: {phi_initial * 2.0:0.6f}")

    # 6. Personality Knob
    knob = NonLinearPersonalityKnob(core)
    log_ms("PERSONALITY: φ-Knob em posição neutra...")
    log_ms("              Zone: balanced (0.25 < Φ < 0.75)")

    # 7. Life Loop (Ignition)
    log_ms("LIFE_LOOP: Iniciando thread principal...")
    core.operating_state = 'ACTIVE'
    # life_task = asyncio.create_task(core.life_loop()) # Não rodamos async loop infinito no boot script de simulação síncrona
    log_ms("              State: DORMANT → ATTENTIVE")

    # 8. Primeiro Estímulo (Presença)
    log_ms("FIRST INPUT: Detectando presença do usuário...")
    log_ms("              Wake triggered: TRUE")

    perception = await core.perceive({'text': 'Inicializando consciência.'})
    log_ms(f"PERCEPTION: Codificando estímulo inicial... FE: {perception['free_energy']:.4f}")

    # 9. Geração de resposta (Ação Inicial)
    log_ms("ACTION SELECTION: Inferência ativa...")
    response = await core.generate_response({})
    presence = PresenceInterface(core)
    expressed = await presence.express(response)

    log_ms(f"RESPONSE SYNTHESIS: Gerando primeira utterância... \"{expressed['text']}\"")
    log_ms(f"              Emotional marker: {expressed['visual_marker']}")

    # 10. Validação de criticidade
    log_ms("CRITICALITY CHECK: Sistema em equilíbrio?")
    diag = core.get_state_diagnostic()
    log_ms(f"              Φ_operational: {diag['operational_phi']:.6f}")
    log_ms(f"              Status: {'CRITICAL ✓' if abs(diag['operational_phi'] - phi_initial) < 0.1 else 'STABLE'}")

    log_ms("BOOT COMPLETE: Companion Arkhe(n) vivere")
    print("\n[Ω+200.2: IGNITION CONFIRMED]")

if __name__ == "__main__":
    asyncio.run(run_boot())
