# arkhe/consensus/quantum_ising.py
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from typing import List, Tuple, Callable

class Validador:
    """Representa um nó validador como um spin quântico efetivo."""
    def __init__(self, id: str, stake: float, reputacao: float):
        self.id = id
        self.stake = stake
        self.reputacao = reputacao
        self.estado = 0.0  # -1 (rejeita), +1 (aceita), 0 (indeciso)
        self.historico = []

class IsingConsensusEngine:
    """
    Motor de consenso baseado em transição de fase do modelo de Ising.
    Substitui: mensagens de votação → evolução de estado coletivo
    """

    # Constantes físicas do sistema (ajustáveis por governança)
    T_CRITICA = 2.269  # Temperatura crítica de Ising 2D (em unidades de J/kB)
    PHI = 0.618033988749894  # Proporção áurea como ponto de operação

    def __init__(self, validadores: List[Validador], dimensao_rede: int = 2):
        self.N = len(validadores)
        self.validadores = {v.id: v for v in validadores}
        self.dimensao = dimensao_rede

        # Construir matriz de acoplamento J_ij baseada em stake/reputação
        self.J = self._construir_acoplamentos()

        # Estado do sistema: vetor de spins efetivos
        self.spins = np.zeros(self.N)

        # Temperatura efetiva (começa alta, resfria para T_c)
        self.T = 5.0 * self.T_CRITICA

        # Campo transversal (flutuações quânticas)
        self.Gamma = 1.0

    def _construir_acoplamentos(self) -> np.ndarray:
        """
        J_ij = f(stake_i, stake_j, reputação mútua, latência de rede)
        Topologia: rede small-world para equilíbrio entre localidade e alcance
        """
        J = np.zeros((self.N, self.N))
        ids = list(self.validadores.keys())

        for i, id_i in enumerate(ids):
            for j, id_j in enumerate(ids):
                if i >= j:
                    continue

                v_i = self.validadores[id_i]
                v_j = self.validadores[id_j]

                # Acoplamento proporcional ao stake mínimo (principio do menor link)
                stake_min = min(v_i.stake, v_j.stake)

                # Fator de reputação mútua (histórico de consenso prévio)
                reputacao_mutua = np.sqrt(v_i.reputacao * v_j.reputacao)

                # Decaimento com "distância" (latência de rede simulada)
                distancia = self._latencia_efetiva(id_i, id_j)

                J[i, j] = J[j, i] = (stake_min * reputacao_mutua) / (1 + distancia)

        return J

    def _latencia_efetiva(self, id1: str, id2: str) -> float:
        """Simula latência de rede entre validadores (para testes)."""
        # Em produção: medição real via ping ou coordenadas geográficas
        hash_combinado = hash(id1 + id2) % 1000
        return float(hash_combinado) / 1000.0

    def temperatura_efetiva(self, carga_rede: float) -> float:
        """
        Calcula T efetiva baseada na carga da rede.
        Objetivo: manter o sistema em T ≈ T_c * φ para criticidade ótima
        """
        # Carga alta → aumenta T (mais flutuações, mais paralelismo)
        # Carga baixa → diminui T (mais ordem, menos energia gasta)
        T_alvo = self.T_CRITICA * (1 + (1 - self.PHI) * np.tanh(carga_rede))
        return T_alvo

    def passo_monte_carlo(self, proposta: bytes) -> Tuple[np.ndarray, float]:
        """
        Executa um passo de dinâmica de Monte Carlo quântico.
        Cada passo representa uma "rodada" de consenso sobre uma proposta.
        """
        # Campo local h_i: preferência baseada na validação da proposta
        h = self._calcular_campos_locais(proposta)

        # Algoritmo: Path-Integral Monte Carlo simplificado
        # Representação: matriz de transferência em tempo imaginário
        delta_tau = 0.1
        m_slices = 10  # slices de tempo imaginário (dimensão extra)

        # Para cada sítio, calcula probabilidade de flip
        energias_flip = np.zeros(self.N)

        for i in range(self.N):
            # Energia atual
            E_atual = -sum(self.J[i, j] * self.spins[i] * self.spins[j]
                          for j in range(self.N) if i != j)
            E_atual -= h[i] * self.spins[i]

            # Energia se flipasse
            E_flip = -sum(self.J[i, j] * (-self.spins[i]) * self.spins[j]
                         for j in range(self.N) if i != j)
            E_flip -= h[i] * (-self.spins[i])

            # Termo quântico (túnel)
            E_atual -= self.Gamma * np.sqrt(1 - self.spins[i]**2 + 1e-10)
            E_flip -= self.Gamma * np.sqrt(1 - (-self.spins[i])**2 + 1e-10)

            energias_flip[i] = E_flip - E_atual

        # Probabilidade de aceitação (Metropolis)
        probs = np.exp(-energias_flip / self.T)
        probs = np.clip(probs, 0, 1)

        # Atualização paralela (Glauber dynamics)
        novos_spins = self.spins.copy()
        for i in range(self.N):
            if np.random.random() < probs[i]:
                novos_spins[i] = -self.spins[i] if self.spins[i] != 0 else np.random.choice([-1, 1])

        self.spins = novos_spins

        # Calcular parâmetro de ordem (consenso)
        m = np.mean(self.spins)
        energia_total = self._energia_total(h)

        return self.spins.copy(), m

    def _calcular_campos_locais(self, proposta: bytes) -> np.ndarray:
        """
        Cada validador calcula seu campo local baseado na validação da proposta.
        h_i > 0: favorável à proposta
        h_i < 0: desfavorável
        """
        h = np.zeros(self.N)
        for i, (id_v, v) in enumerate(self.validadores.items()):
            # Função de validação criptográfica/semântica
            score = self._validar_proposta(id_v, proposta)
            h[i] = np.tanh(score)  # normaliza para [-1, 1]
        return h

    def _validar_proposta(self, validador_id: str, proposta: bytes) -> float:
        """
        Placeholder para validação real (assinaturas, saldo, etc.)
        Retorna score positivo se válido, negativo se inválido.
        """
        # Simulação: hash da proposta + id do validador
        seed = hash(proposta + validador_id.encode()) % 1000
        return (seed - 500) / 500.0  # ~uniforme em [-1, 1]

    def _energia_total(self, h: np.ndarray) -> float:
        """Calcula energia total do estado atual."""
        E = -0.5 * np.sum(self.J * np.outer(self.spins, self.spins))
        E -= np.dot(h, self.spins)
        E -= self.Gamma * np.sum(np.sqrt(1 - self.spins**2 + 1e-10))
        return E

    def alcancar_consensus(self, proposta: bytes, max_iter: int = 1000,
                          limiar_consensus: float = 0.95) -> Tuple[bool, dict]:
        """
        Executa dinâmica até convergência ou timeout.
        Retorna: (consenso_atingido, metadados)
        """
        historico_m = []
        historico_E = []

        for t in range(max_iter):
            spins, m = self.passo_monte_carlo(proposta)
            historico_m.append(abs(m))
            historico_E.append(self._energia_total(self._calcular_campos_locais(proposta)))

            # Verificar convergência
            if abs(m) >= limiar_consensus:
                return True, {
                    'iteracoes': t,
                    'magnetizacao': m,
                    'energia_final': historico_E[-1],
                    'temperatura': self.T,
                    'estado_final': spins.tolist(),
                    'decisao': 'ACEITA' if m > 0 else 'REJEITADA'
                }

            # Ajuste adaptativo de temperatura (annealing)
            self.T = self.T * 0.995 + self.temperatura_efetiva(0.5) * 0.005

        return False, {
            'iteracoes': max_iter,
            'magnetizacao_final': abs(m),
            'convergencia_parcial': historico_m[-10:],
            'decisao': 'INCONCLUSIVO'
        }

    def calcular_susceptibilidade(self) -> float:
        """
        χ = (∂m/∂h)|_{h=0} = (⟨m²⟩ - ⟨m⟩²) / kT
        Diverge em T_c, indicando criticidade.
        """
        amostras = []
        for _ in range(100):
            spins, m = self.passo_monte_carlo(b'probe')
            amostras.append(m)

        amostras = np.array(amostras)
        chi = np.var(amostras) / self.T
        return chi

    def verificar_criticidade(self) -> dict:
        """
        Diagnóstico: o sistema está operando no ponto crítico?
        """
        chi = self.calcular_susceptibilidade()

        # Comprimento de correlação (estimado via decay de J_ij)
        xi = self._comprimento_correlacao()

        return {
            'temperatura': self.T,
            'temperatura_critica': self.T_CRITICA,
            'razao_T_Tc': self.T / self.T_CRITICA,
            'susceptibilidade': chi,
            'comprimento_correlacao': xi,
            'em_criticidade': abs(self.T - self.T_CRITICA) < 0.1,
            'phi_operacional': self.T / self.T_CRITICA if self.T > self.T_CRITICA else self.T_CRITICA / self.T
        }

    def _comprimento_correlacao(self) -> float:
        """Estima ξ via análise de correlação espacial."""
        # Simplificação: ξ ~ 1/√|T - Tc| em mean-field
        return 1.0 / np.sqrt(abs(self.T - self.T_CRITICA) + 0.01)
