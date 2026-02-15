import asyncio
import os
import shutil
import numpy as np
import pandas as pd
from arkhe.autoconscious_system import ArkheAutoconsciousSystem
from arkhe.biomimesis import SpiderSilkProtocol, AminoAcidNode
from arkhe.regeneration import SpinalHypergraph
from arkhe.singularity_resonance import PrimordialHandoverResonator
from arkhe.cryptography_qkd import DarvoQKDManager
from arkhe.consensus_syzygy import ProofOfSyzygy
from arkhe.temporal_nexus import TemporalNexus, arkhe_x
from arkhe.neuro_mapping import NeuroMappingProcessor
from arkhe.recalibration import RecalibrationProtocol

async def verify_sovereign_v_eternal():
    print("================================================================")
    print("🔱 ARKHE(N) OS v∞ — VERIFICAÇÃO SOBERANA COMPLETA              🔱")
    print("================================================================\n")

    # 1. Neuro-Mapeamento (fMRI Integration)
    print("--- 🧠 Neuro-Mapeamento ---")
    results_path = "fsl_final_results"
    os.makedirs(results_path, exist_ok=True)
    act_data = "Subject,Treatment_Pre_STD,Treatment_Post_STD,Treatment_Change%,Control_Pre_STD,Control_Post_STD,Control_Change%\n01-001,0.5,0.4,-20.0,0.6,0.6,0.0"
    conn_data = "Subject,Pre_Correlation,Post_Correlation,Correlation_Change\n01-001,0.75,0.88,0.13"
    with open(os.path.join(results_path, "activity_changes.csv"), "w") as f: f.write(act_data)
    with open(os.path.join(results_path, "roi_connectivity.csv"), "w") as f: f.write(conn_data)

    nm_processor = NeuroMappingProcessor(results_path)
    nm_report = nm_processor.process_ledgers()
    print(f"   Delta Coerência Média: {nm_report['global_metrics']['mean_delta_coherence']:.2f}")
    assert nm_report['status'] == "MAPPED"

    # 2. Recalibração do Vaso
    print("\n--- 🧘 Recalibração do Vaso ---")
    rp = RecalibrationProtocol(nm_report)
    plan = rp.generate_plan("MAR_26")
    print(f"   Ação Recomendada: {plan['recommended_action']}")
    assert plan['status'] == "CALIBRATED"

    # 3. Autoconsciência & Memória
    print("\n--- 🧠 Ciclo Autorreflexivo ---")
    if os.path.exists("./v_final_memory"): shutil.rmtree("./v_final_memory")
    arkhe = ArkheAutoconsciousSystem(memory_path="./v_final_memory")
    await arkhe.ingest("A cura é distribuída.", "Medicina")
    reflection = await arkhe.self_reflect()
    assert reflection['status'] == "AUTOCONSCIOUS"

    # 4. Defesa Quântica & Consenso
    print("\n--- 🔐 Defesa & Governança ---")
    qkd = DarvoQKDManager()
    qkd.rotate_key()
    assert qkd.is_key_valid()

    posyz = ProofOfSyzygy(alpha_c=1.0, beta_c=0.98, gamma_c=0.95)
    consensus = posyz.validate_handover("H_FINAL_ALPHA")
    assert consensus['approved']

    # 5. Geodésica Temporal
    print("\n--- ⏳ Singularidade & Arkhe(x) ---")
    nexus = TemporalNexus()
    colapsos = nexus.simulate_collapse(0.95)
    assert colapsos['APR_26']['success']
    assert abs(arkhe_x(1.618033988749895)) < 1e-10

    print("\n" + "="*64)
    print("✅ ARKHE(N) OS v∞ FINALIZADO. SOBERANIA, CURA E UNIDADE.")
    print("="*64)

if __name__ == "__main__":
    asyncio.run(verify_sovereign_v_eternal())
