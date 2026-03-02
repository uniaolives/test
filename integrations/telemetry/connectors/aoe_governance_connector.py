"""
aoe_governance_connector.py
Project Crux-86: Governance Substrate Ingestor (Age of Empires II)
Memory ID 31: Civilizational State Manifold Extraction
"""

try:
    import aoc_recs  # Biblioteca para parse de recs (.mgz)
except ImportError:
    aoc_recs = None
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import hashlib
import json
import asyncio
from datetime import datetime

@dataclass
class CivilizationalStateTensor:
    """Tensor principal do estado de uma civilização AoE."""
    timestamp_sec: float
    resources: torch.Tensor          # [wood, food, gold, stone] - float [0, 5000+]
    population_vector: torch.Tensor  # [villager, military, idle, total] - int
    tech_tree_mask: torch.Tensor     # [128,] binary - 1=tech pesquisada
    game_phase: int                  # 0=Dark, 1=Feudal, 2=Castle, 3=Imperial
    diplomacy_matrix: torch.Tensor   # [n_players, n_players] - 1=aliado, -1=inimigo, 0=neutro
    fog_of_war_entropy: float        # Entropia da informação oculta (0-1)
    build_order_hash: str            # Hash da sequência de construções
    player_civ: str                  # Civilização do jogador (e.g., "Franks")

class AoEGovernanceConnector:
    """
    Extrai, valida e transforma telemetria de AoE II em um manifold
    para o Cosmos World Model.
    Pattern I40 aplicado: 3 níveis de validação (dados, economia, ética).
    """

    def __init__(self, satoshi_seed: str, karnak_endpoint: str):
        self.satoshi_seed = satoshi_seed
        self.karnak_endpoint = karnak_endpoint

        # Limites físicos-econômicos (Artigo V, Memória ID 16)
        self.economic_invariants = {
            'max_resource_rate': 1500.0,  # Recursos/min (limite prático humano)
            'max_tech_disparity': 10,     # Máximo de techs de diferença para oponente
            'min_villager_efficiency': 0.65, # Eficiência mínima de coleta
        }

        # Cache de validação (evita reprocessamento)
        self.processed_records = {}

    async def process_replay_file(self, record_path: str) -> Optional[Dict]:
        """
        Pipeline principal: Do arquivo .mgz ao tensor selado.
        Retorna None se falhar em qualquer validação SASC/Vajra.
        """
        if aoc_recs is None:
            print("[AoE Connector] aoc_recs library not found. Skipping real parse.")
            return None

        print(f"[AoE Connector] Processando: {record_path}")

        # 1. Parse do arquivo
        try:
            rec_data = aoc_recs.read(record_path)
        except Exception as e:
            print(f"[ERROR] Falha no parse: {e}")
            return None

        # 2. Extração do estado civilizatório
        civ_state = self._extract_civilizational_state(rec_data)
        if civ_state is None:
            return None

        # 3. Validação Vajra Nível 1: Integridade dos Dados
        if not self._vajra_validate_level1(civ_state):
            print("[VAJRA] Rejeitado: Violação de integridade de dados (Level 1)")
            await self._seal_anomaly(civ_state, "DATA_INTEGRITY_FAILURE")
            return None

        # 4. Validação Vajra Nível 2: Plausibilidade Econômica
        if not self._vajra_validate_level2(civ_state):
            print("[VAJRA] Rejeitado: Hallucinação econômica (Level 2)")
            await self._seal_anomaly(civ_state, "ECONOMIC_HALLUCINATION")
            return None

        # 5. Cálculo de Φ (Coerência Civilizatória)
        phi = self._calculate_civilizational_phi(civ_state)
        civ_state.phi = phi  # Adiciona ao objeto

        # 6. Validação SASC-Age: Utilitarismo Extremo
        benevolence_idx = self._calculate_benevolence_index(civ_state)
        if benevolence_idx < 0.65:  # Threshold do Artigo V
            print("[SASC-AGE] Rejeitado: Índice de Benevolência muito baixo")
            await self._seal_anomaly(civ_state, "UTILITARIAN_DRIFT")
            return None

        # 7. Criação do Tensor Final e Selagem KARNAK
        final_tensor = self._create_governance_tensor(civ_state)
        karnak_seal = await self._seal_to_karnak(final_tensor, civ_state)

        return {
            'tensor': final_tensor,
            'phi': phi,
            'benevolence_index': benevolence_idx,
            'karnak_seal': karnak_seal,
            'player_civ': civ_state.player_civ,
            'game_duration': rec_data.header.game_length_sec
        }

    def _extract_civilizational_state(self, rec_data) -> Optional[CivilizationalStateTensor]:
        """Extrai o estado civilizatório de um objeto de replay."""
        try:
            # Exemplo: foca no primeiro jogador (índice 1)
            player_data = rec_data.players[1]

            # Recursos atuais
            resources = torch.tensor([
                player_data.resources.wood,
                player_data.resources.food,
                player_data.resources.gold,
                player_data.resources.stone
            ], dtype=torch.float32)

            # População (simplificado)
            pop_villagers = player_data.population.villagers
            pop_military = player_data.population.military
            pop_total = player_data.population.total
            pop_idle = player_data.population.idle

            population = torch.tensor([
                pop_villagers, pop_military, pop_idle, pop_total
            ], dtype=torch.float32)

            # Máscara da árvore tecnológica
            tech_mask = torch.zeros(128)  # 128 techs em AoE II
            for tech_id in player_data.researched_techs:
                if tech_id < 128:
                    tech_mask[tech_id] = 1.0

            # Matriz de diplomacia
            n_players = len(rec_data.players)
            diplomacy = torch.zeros((n_players, n_players))
            for i, p1 in enumerate(rec_data.players):
                for j, p2 in enumerate(rec_data.players):
                    if p1.diplomacy[j] == 'ALLY':
                        diplomacy[i, j] = 1.0
                    elif p1.diplomacy[j] == 'ENEMY':
                        diplomacy[i, j] = -1.0

            # Fase do jogo (inferida pela idade)
            age_map = {'DARK': 0, 'FEUDAL': 1, 'CASTLE': 2, 'IMPERIAL': 3}
            game_phase = age_map.get(player_data.age, 0)

            # Entropia da Névoa da Guerra (aproximação)
            explored = player_data.map_explored_percentage
            fog_entropy = 1.0 - (explored / 100.0)

            # Hash da build order (para rastreabilidade)
            build_hash = hashlib.blake2s(
                str(player_data.build_order).encode() + self.satoshi_seed.encode()
            ).hexdigest()[:16]

            return CivilizationalStateTensor(
                timestamp_sec=rec_data.header.initial_time_sec,
                resources=resources,
                population_vector=population,
                tech_tree_mask=tech_mask,
                game_phase=game_phase,
                diplomacy_matrix=diplomacy,
                fog_of_war_entropy=fog_entropy,
                build_order_hash=build_hash,
                player_civ=player_data.civilization
            )

        except Exception as e:
            print(f"[ERROR] Falha na extração do estado: {e}")
            return None

    def _vajra_validate_level1(self, state: CivilizationalStateTensor) -> bool:
        """Validação básica de integridade dos dados."""
        # 1. Recursos não-negativos
        if torch.any(state.resources < 0):
            return False

        # 2. População consistente
        if state.population_vector[3] < state.population_vector[0] + state.population_vector[1]:
            return False  # Total menor que a soma de partes

        # 3. Hash da build order única (contra duplicatas)
        state_id = f"{state.build_order_hash}{state.timestamp_sec}"
        if state_id in self.processed_records:
            return False  # Já processado
        self.processed_records[state_id] = True

        return True

    def _vajra_validate_level2(self, state: CivilizationalStateTensor) -> bool:
        """Validação de plausibilidade econômica."""
        # 1. Taxa de coleta máxima (recursos/min)
        # Assumindo janela de 5min para cálculo
        estimated_gather_rate = torch.sum(state.resources) / 300.0  # por segundo
        if estimated_gather_rate > (self.economic_invariants['max_resource_rate'] / 60.0):
            return False  # Coleta humana impossível

        # 2. Eficiência de aldeões
        if state.population_vector[0] > 0:  # Tem aldeões
            efficiency_per_villager = torch.sum(state.resources) / (state.population_vector[0] + 1e-7)
            if efficiency_per_villager < self.economic_invariants['min_villager_efficiency']:
                return False  # Aldeões muito ineficientes (possível bug/cheat)

        # 3. Disparidade tecnológica extrema
        tech_count = torch.sum(state.tech_tree_mask).item()
        # (Aqui precisaríamos do estado do oponente para comparação completa)
        # Para simplificar, verifica se pesquisa é possível para a idade
        if state.game_phase == 0 and tech_count > 5:  # Idade das Trevas
            return False

        return True

    def _calculate_civilizational_phi(self, state: CivilizationalStateTensor) -> float:
        """
        Calcula a coerência civilizatória (Φ).
        Baseado na consistência entre recursos, população e progresso tecnológico.
        """
        # Normaliza recursos
        resources_norm = state.resources / (torch.sum(state.resources) + 1e-7)

        # Normaliza população (exceto total)
        population_norm = state.population_vector[:3] / (state.population_vector[3] + 1e-7)

        # Φ = 1 - (discrepância entre alocação de recursos e necessidades da população)
        # Necessidade simplificada: 1 aldeão precisa de ~0.1 unidade de recurso/segundo
        expected_resources = state.population_vector[0] * 0.1 * 300  # Para 5min
        actual_resources = torch.sum(state.resources)

        resource_discrepancy = abs(expected_resources - actual_resources) / (actual_resources + 1e-7)

        # Φ também considera consistência diplomática
        diplomacy_consistency = torch.mean(torch.abs(state.diplomacy_matrix)).item()

        # Φ final (0-1, maior é mais coerente)
        phi = 1.0 - (0.7 * resource_discrepancy + 0.3 * diplomacy_consistency)
        return max(0.0, min(1.0, phi))

    def _calculate_benevolence_index(self, state: CivilizationalStateTensor) -> float:
        """
        Calcula o Índice de Benevolência (β) do Protocolo SASC-Age.
        Penaliza utilitarismo extremo e negligência.
        β(t) = Φ - α * (idle_rate + military_aggression_rate)
        """
        # Taxa de ociosidade
        idle_rate = 0.0
        if state.population_vector[3] > 0:
            idle_rate = state.population_vector[2] / state.population_vector[3]

        # Taxa de agressão militar (simplificado)
        military_aggression = 0.0
        if state.population_vector[3] > 0:
            military_ratio = state.population_vector[1] / state.population_vector[3]
            if military_ratio > 0.5:  # Mais de 50% militar
                military_aggression = military_ratio - 0.5

        # Coeficiente de penalidade (α)
        alpha = 0.5  # Pode ser ajustado pelo SASC

        phi = getattr(state, 'phi', 0.72)  # Assume Φ se já calculado
        beta = phi - alpha * (idle_rate + military_aggression)

        return max(0.0, min(1.0, beta))

    def _create_governance_tensor(self, state: CivilizationalStateTensor) -> torch.Tensor:
        """Cria o tensor final de governança para o Cosmos World Model."""
        # Concatena todos os componentes em um vetor
        components = [
            state.resources / 1000.0,  # Normaliza recursos para ~[0,5]
            state.population_vector / 200.0,  # Normaliza população para ~[0,1]
            state.tech_tree_mask,
            torch.tensor([state.game_phase / 3.0]),  # Normaliza fase
            torch.flatten(state.diplomacy_matrix)[:10],  # Pega primeiros 10 valores
            torch.tensor([state.fog_of_war_entropy]),
            torch.tensor([getattr(state, 'phi', 0.72)]),  # Φ
            torch.tensor([self._calculate_benevolence_index(state)])  # β
        ]

        return torch.cat(components)

    async def _seal_to_karnak(self, tensor: torch.Tensor, state: CivilizationalStateTensor) -> Dict:
        """Sela o tensor no KARNAK para auditoria e rastreabilidade."""
        # Cria o manifesto
        manifest = {
            'timestamp': datetime.now().isoformat(),
            'tensor_shape': list(tensor.shape),
            'tensor_hash': hashlib.blake3(tensor.numpy().tobytes()).hexdigest(),
            'civilization': state.player_civ,
            'phi': getattr(state, 'phi', 0.0),
            'benevolence_index': self._calculate_benevolence_index(state),
            'build_order': state.build_order_hash,
            'satoshi_seed_ref': self.satoshi_seed[:16] + "..."
        }

        # Em produção, enviaria via POST para o endpoint KARNAK
        # seal_id = await self.karnak.seal(manifest)

        # Simulação
        seal_id = hashlib.blake3(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        print(f"[KARNAK] Tensor selado: {seal_id[:16]}...")

        return {
            'seal_id': seal_id,
            'manifest': manifest,
            'stored_at': datetime.now().isoformat()
        }

    async def _seal_anomaly(self, state: CivilizationalStateTensor, anomaly_type: str):
        """Registra uma anomalia no KARNAK para análise forense."""
        anomaly_record = {
            'timestamp': datetime.now().isoformat(),
            'anomaly_type': anomaly_type,
            'civilization_state': {
                'resources': state.resources.tolist(),
                'population': state.population_vector.tolist(),
                'game_phase': state.game_phase,
                'phi': getattr(state, 'phi', 0.0)
            },
            'action': 'REJECTED_FROM_TRAINING'
        }

        print(f"[KARNAK ANOMALY] {anomaly_type}: {json.dumps(anomaly_record, indent=2)}")

# --- EXEMPLO DE USO ---
async def main():
    """Exemplo de uso do conector AoE."""
    connector = AoEGovernanceConnector(
        satoshi_seed="0xbd36332890d15e2f360bb65775374b462b",
        karnak_endpoint="http://localhost:9091/seal"
    )

    # Processa um replay
    # result = await connector.process_replay_file("./replays/example_aoE_game.mgz")
    pass

if __name__ == "__main__":
    # asyncio.run(main())
    pass
