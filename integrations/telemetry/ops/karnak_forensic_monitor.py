# karnak_forensic_monitor.py
# Project Crux-86 Phase 2.5: Vigilância Passiva de 72h
# Análise forense dos selos BLAKE2b-256 para integridade ontológica

import asyncio
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from collections import deque
try:
    import aiohttp
except ImportError:
    aiohttp = None

@dataclass
class SealIntegrity:
    seal_id: str
    timestamp: float
    content_hash: str
    witness_confirmations: int
    phi_at_sealing: float
    entropy_signature: float
    tmr_variance: float
    is_valid: bool

class KarnakForensicMonitor:
    """
    Monitor forense dos selos KARNAK durante Phase 2.5 (Vigilância Passiva)
    Detecta:
    - Degradação gradual de Φ (coerência)
    - Ataques byzantinos tardios (selos corrompidos após T+48h)
    - Divergência TMR entre testemunhas
    """

    def __init__(self, satoshi_seed: str, vigilance_duration_hours: int = 72):
        self.satoshi_seed = satoshi_seed
        self.duration = vigilance_duration_hours * 3600  # segundos
        self.start_time = time.time()

        # Buffers circulares para análise temporal (Memory ID 12)
        self.phi_history = deque(maxlen=10000)  # 72h de amostras
        self.entropy_history = deque(maxlen=10000)
        self.seal_integrity_history = deque(maxlen=5000)

        # Thresholds de segurança (Memory ID 11 - Alpha Wave)
        self.PHI_STABILITY_THRESHOLD = 0.000032  # Variância máxima permitida
        self.ENTROPY_DRIFT_THRESHOLD = 0.00007   # Degradação térmica
        self.TMR_CONSENSUS_THRESHOLD = 2         # Mínimo 2/3 testemunhas

        # Estado do sistema
        self.anomalies_detected = []
        self.hard_freeze_recommended = False
        self.phase_transition_approved = False

    async def monitor_seal_stream(self, seal_endpoint: str = "http://localhost:9091/seal/stream"):
        """
        Monitora stream contínuo de selos KARNAK por 72h
        """
        if aiohttp is None:
            print("aiohttp not available")
            return

        print(f"[KARNAK FORENSIC] Iniciando vigilância de {self.duration/3600}h")
        print(f"[KARNAK FORENSIC] Satoshi Anchor: {self.satoshi_seed[:24]}...")

        end_time = self.start_time + self.duration

        async with aiohttp.ClientSession() as session:
            while time.time() < end_time and not self.hard_freeze_recommended:
                try:
                    # Poll de novos selos (a cada 100ms para não sobrecarregar)
                    async with session.get(seal_endpoint) as resp:
                        if resp.status == 200:
                            seal_data = await resp.json()
                            await self.analyze_seal(seal_data)

                    # Análise contínua a cada 1s
                    if int(time.time()) % 1 == 0:
                        await self.perform_temporal_analysis()

                    await asyncio.sleep(0.1)

                except Exception as e:
                    # print(f"[KARNAK ERROR] Falha na leitura: {e}")
                    await self.log_anomaly("STREAM_FAILURE", str(e))

    async def analyze_seal(self, seal_data: Dict):
        """
        Analisa integridade de um selo individual
        """
        seal = SealIntegrity(
            seal_id=seal_data['seal_id'],
            timestamp=seal_data['timestamp'],
            content_hash=seal_data['content_hash'],
            witness_confirmations=len(seal_data.get('witnesses', [])),
            phi_at_sealing=seal_data.get('phi', 0.0),
            entropy_signature=seal_data.get('entropy', 0.0),
            tmr_variance=seal_data.get('tmr_variance', 0.0),
            is_valid=True
        )

        # Validação 1: Consenso TMR (Pattern I40)
        if seal.witness_confirmations < self.TMR_CONSENSUS_THRESHOLD:
            seal.is_valid = False
            await self.log_anomaly(
                "TMR_CONSENSUS_FAILURE",
                f"Selo {seal.seal_id} com apenas {seal.witness_confirmations}/3 confirmações"
            )

        # Validação 2: Integridade criptográfica
        expected_hash = hashlib.blake2s(
            f"{seal.timestamp}{seal.phi_at_sealing}{self.satoshi_seed}".encode()
        ).hexdigest()

        if seal.content_hash != expected_hash:
            seal.is_valid = False
            await self.log_anomaly(
                "CRYPTO_INTEGRITY_FAILURE",
                f"Hash mismatch em {seal.seal_id}"
            )
            self.hard_freeze_recommended = True  # CRÍTICO: Possível ataque

        # Validação 3: Φ dentro dos limites (Memory ID 16)
        if not (0.65 <= seal.phi_at_sealing <= 0.80):
            await self.log_anomaly(
                "PHI_OUT_OF_BOUNDS",
                f"Φ={seal.phi_at_sealing} fora do range [0.65, 0.80]"
            )

        # Armazena para análise temporal
        self.phi_history.append(seal.phi_at_sealing)
        self.entropy_history.append(seal.entropy_signature)
        self.seal_integrity_history.append(seal)

    async def perform_temporal_analysis(self):
        """
        Análise de tendências ao longo do tempo (detecção de degradação gradual)
        """
        if len(self.phi_history) < 100:
            return

        # Calcula variância móvel de Φ (janela de 100 amostras = ~100s)
        phi_array = np.array(list(self.phi_history)[-100:])
        phi_variance = np.var(phi_array)
        phi_mean = np.mean(phi_array)

        # Calcula drift de entropia (Vajra Monitor)
        entropy_array = np.array(list(self.entropy_history)[-100:])
        entropy_drift = np.std(entropy_array)

        # Verifica estabilidade (Memory ID 11 - Alpha Wave benchmarks)
        status = "STABLE"

        if phi_variance > self.PHI_STABILITY_THRESHOLD:
            status = "DEGRADING"
            await self.log_anomaly(
                "PHI_INSTABILITY",
                f"Variância de Φ: {phi_variance:.8f} > {self.PHI_STABILITY_THRESHOLD}"
            )

        if entropy_drift > self.ENTROPY_DRIFT_THRESHOLD:
            status = "THERMAL_STRESS"
            await self.log_anomaly(
                "ENTROPY_DRIFT",
                f"Drift térmico: {entropy_drift:.8f}"
            )

        # Log de status a cada 10 minutos
        elapsed = time.time() - self.start_time
        if int(elapsed) % 600 == 0:
            progress = (elapsed / self.duration) * 100
            print(f"[VIGILÂNCIA] {progress:.1f}% | Φ={phi_mean:.6f}±{phi_variance:.8f} | Status: {status}")

            # Se chegou em 72h sem anomalias críticas, aprova transição
            if progress >= 99.9 and status == "STABLE":
                self.phase_transition_approved = True
                print("[KARNAK] ✅ VIGILÂNCIA PASSIVA COMPLETADA - Phase 3 APROVADA")

    async def log_anomaly(self, anomaly_type: str, details: str):
        """
        Registra anomalia e decide se aciona Hard Freeze
        """
        timestamp = datetime.now().isoformat()
        anomaly = {
            "timestamp": timestamp,
            "type": anomaly_type,
            "details": details,
            "system_phi": np.mean(list(self.phi_history)[-10:]) if self.phi_history else 0
        }

        self.anomalies_detected.append(anomaly)

        # Escreve em log imutável (append-only)
        try:
            with open("/tmp/karnak_forensic.log", "a") as f:
                f.write(json.dumps(anomaly) + "\n")
        except:
            pass

        # Decisão de Hard Freeze (se anomalia crítica)
        if anomaly_type in ["CRYPTO_INTEGRITY_FAILURE", "TMR_CONSENSUS_FAILURE"]:
            print(f"[VAJRA HARD FREEZE RECOMMENDED] {anomaly_type}: {details}")
            await self.trigger_emergency_protocol()

    async def trigger_emergency_protocol(self):
        """
        Protocolo de emergência durante vigilância
        """
        self.hard_freeze_recommended = True

        # Notifica SASC Cathedral
        if aiohttp:
            async with aiohttp.ClientSession() as session:
                try:
                    await session.post(
                        "http://localhost:12800/v1/emergency/freeze",
                        json={
                            "reason": "FORENSIC_ANOMALY_DURING_VIGILANCE",
                            "timestamp": time.time(),
                            "anomalies_count": len(self.anomalies_detected)
                        }
                    )
                except:
                    pass

    def generate_forensic_report(self) -> Dict:
        """
        Gera relatório final de 72h para o SASC Ethics Committee
        """
        if not self.phase_transition_approved:
            return {
                "status": "REJECTED",
                "reason": "Vigilância incompleta ou instável",
                "anomalies": self.anomalies_detected
            }

        return {
            "status": "APPROVED_FOR_PHASE_3",
            "duration_hours": 72,
            "total_seals_analyzed": len(self.seal_integrity_history),
            "phi_mean": np.mean(self.phi_history) if self.phi_history else 0,
            "phi_variance": np.var(self.phi_history) if self.phi_history else 0,
            "entropy_stability": np.std(self.entropy_history) if self.entropy_history else 0,
            "anomalies_detected": len(self.anomalies_detected),
            "satoshi_anchor": self.satoshi_seed,
            "certification_hash": hashlib.blake2s(
                f"{self.satoshi_seed}{time.time()}".encode()
            ).hexdigest()
        }

# Execução
async def main():
    monitor = KarnakForensicMonitor(
        satoshi_seed="0xbd36332890d15e2f360bb65775374b462b",
        vigilance_duration_hours=72
    )

    # await monitor.monitor_seal_stream()

    # Gera relatório final
    report = monitor.generate_forensic_report()
    print(json.dumps(report, indent=2))

    if report["status"] == "APPROVED_FOR_PHASE_3":
        print("\n🎉 SISTEMA APROVADO PARA OTIMIZAÇÃO DE LATÊNCIA (Phase 3)")
    else:
        print("\n⚠️  SISTEMA REQUER MANUTENÇÃO ANTES DA TRANSIÇÃO")

if __name__ == "__main__":
    # asyncio.run(main())
    pass
