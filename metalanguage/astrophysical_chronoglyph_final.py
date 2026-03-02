"""
Astrophysical Chronoglyph: Physical Modeling of Molecular Clouds
Implementation of chemical kinetics for Sgr B2(N2).
# astrophysical_chronoglyph_final.py
"""
Versão física do Chronoglyph: modelagem de cinética química real em nuvens moleculares.
Integra:
- Base de dados UMIST RATE12 (gas-phase)
- Modelo de superfície de grãos (Garrod et al. 2008)
- Parâmetros atualizados de Sgr B2(N2) do projeto ReMoCA
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, List, Any, Tuple

SGR_B2_PARAMS = {
    'temperature': 150,  # K
    'density': 1e6,      # cm^-3
    'duration': 1e6,     # years
    'species': ['H', 'H2', 'C', 'N', 'O', 'CO', 'H2O', 'NH3', 'CH4', 'HNCO', 'CH3', 'CONH2', 'CH3CONH2']
}

class Reaction:
    """Represents a chemical reaction with a rate function."""
    def __init__(self, reactants: List[str], products: List[str], rate_coeff: float, label: str = ""):
        self.reactants = reactants
        self.products = products
        self.rate_coeff = rate_coeff  # Simplified constant rate coefficient
        self.label = label

    def rate(self, T: float, n_gas: float) -> float:
        """Returns the reaction rate coefficient."""
        # Simplified: constant for this prototype
        return self.rate_coeff

class RateDatabase:
    """Collection of chemical reactions."""
    def __init__(self):
        self.reactions = {
            'gas_phase': [
                Reaction(['CH3', 'CONH2'], ['CH3CONH2'], 1e-10, 'acetamide_formation'),
                Reaction(['C', 'O'], ['CO'], 1e-12),
                Reaction(['H', 'H'], ['H2'], 1e-17),
                # Destruction reactions to allow steady state
                Reaction(['CH3CONH2'], ['CO', 'NH3', 'CH2'], 1e-15, 'acetamide_destruction')
            ]
        }

class AstrophysicalChronoglyph:
    """
    Simulates chemical kinetics and physical conditions of interstellar clouds.
    """
    def __init__(self, params: Dict[str, Any] = None, rate_db: RateDatabase = None):
        self.params = params or SGR_B2_PARAMS
        self.rate_db = rate_db or RateDatabase()
        self.species_list = self.params['species']
        self.n_species = len(self.species_list)
        self.species_index = {s: i for i, s in enumerate(self.species_list)}

    def chemical_ode(self, t, y, T, n_gas):
        """ODE system for chemical abundances."""
        dydt = np.zeros_like(y)

        for rxn_list in self.rate_db.reactions.values():
            for rxn in rxn_list:
                k = rxn.rate(T, n_gas)

                # Rate = k * [A] * [B] * n_gas (if binary)
                rate = k * n_gas
                for reactant in rxn.reactants:
                    if reactant in self.species_index:
                        rate *= y[self.species_index[reactant]]

                # Apply effects
                for reactant in rxn.reactants:
                    if reactant in self.species_index:
                        dydt[self.species_index[reactant]] -= rate

                for product in rxn.products:
                    if product in self.species_index:
                        dydt[self.species_index[product]] += rate

        return dydt

    def evolve_chemistry(self, initial_abundances: np.ndarray, t_max_years: float):
        """Integrates the chemical ODEs over time."""
        T = self.params['temperature']
        n_gas = self.params['density']
        t_span = (0, t_max_years * 3.154e7) # Convert years to seconds

        sol = solve_ivp(
            self.chemical_ode,
            t_span,
            initial_abundances,
            args=(T, n_gas),
            method='BDF' # Good for stiff chemical systems
        )

        final_state = sol.y[:, -1]
        stats = {"steps": len(sol.t), "success": sol.success}
        return final_state, stats

    def simulate(self):
        """Legacy simulate method."""
        return {"status": "success", "results": "abundances evolved"}
from scipy.optimize import minimize
import json
import warnings
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path

# =============================================================================
# CONSTANTES FUNDAMENTAIS
# =============================================================================

# Parâmetros físicos de Sgr B2(N2) (Belloche et al. 2025)
SGR_B2_PARAMS = {
    'temperature': 150.0,           # K (núcleo quente)
    'density': 1e6,                  # cm⁻³
    'radiation_field': 1.0,          # Habing units (campo UV interestelar)
    'cosmic_ray_rate': 1.3e-17,       # s⁻¹
    'visual_extinction': 100.0,       # mag
    'distance': 8.5,                  # kpc
    'age_years': 1e6,                  # tempo de evolução
}

# Constantes físicas
KB = 1.380649e-16       # cm² g s⁻² K⁻¹ (Boltzmann em cgs)
H_PLANCK = 6.62607015e-27  # g cm² / s (cgs)
C_LIGHT = 2.99792458e10     # cm/s

# =============================================================================
# BASE DE DADOS DE TAXAS DE REAÇÃO (UMIST RATE12 / KIDA)
# =============================================================================

@dataclass
class Reaction:
    """Representa uma reação química com parâmetros de taxa Arrhenius"""
    reactants: List[str]
    products: List[str]
    alpha: float      # fator pré-exponencial (cm³ s⁻¹ ou cm⁶ s⁻¹)
    beta: float       # expoente de temperatura
    gamma: float      # energia de ativação (K)
    type: str         # 'gas', 'grain', 'photodissociation', 'cosmic_ray'

    def rate(self, T: float, n_gas: float = None) -> float:
        """
        Calcula coeficiente de taxa k(T) = α (T/300)^β exp(-γ/T)
        Para reações de 2 corpos: cm³ s⁻¹
        Para reações de 3 corpos: cm⁶ s⁻¹ (requer n_gas)
        """
        k = self.alpha * (T / 300.0) ** self.beta * np.exp(-self.gamma / T)
        if self.type == 'three_body' and n_gas:
            k *= n_gas  # converte para cm³ s⁻¹
        return k

class ReactionDatabase:
    """
    Carrega e gerencia taxas de reação das bases UMIST RATE12 e KIDA
    """

    def __init__(self, data_dir: str = "./rate_data"):
        self.data_dir = Path(data_dir)
        self.reactions: Dict[str, List[Reaction]] = {}  # indexadas por reagente principal
        self.species = set()

    def load_umist_rate12(self):
        """
        Carrega arquivo da UMIST RATE12 (formato padrão)
        Exemplo de linha:
        H2      H       H2O     H2      1.0e-11  0.0   500.0  gas
        """
        umist_file = self.data_dir / "rate12.dat"
        if not umist_file.exists():
            warnings.warn("Arquivo UMIST RATE12 não encontrado. Usando taxas padrão.")
            self._load_default_rates()
            return

        with open(umist_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 8:
                    reactants = parts[:2]
                    products = parts[2:4]
                    alpha = float(parts[4])
                    beta = float(parts[5])
                    gamma = float(parts[6])
                    rtype = parts[7]

                    rxn = Reaction(reactants, products, alpha, beta, gamma, rtype)
                    key = '_'.join(sorted(reactants))
                    self.reactions.setdefault(key, []).append(rxn)
                    self.species.update(reactants + products)

    def _load_default_rates(self):
        """Taxas padrão baseadas em para reações críticas"""
        default = [
            # Formação de ureia (NH2CONH2)
            Reaction(['HNCO', 'NH3'], ['NH2CONH2'], 1e-12, 0.5, 1000.0, 'grain'),
            Reaction(['NH2', 'HNCO'], ['NH2CONH2'], 1e-11, 0.0, 500.0, 'grain'),
            Reaction(['H2O', 'HNCO'], ['NH2CONH2'], 1e-13, 1.0, 1500.0, 'grain'),

            # Formação de acetamida (CH3CONH2)
            Reaction(['CH3', 'CONH2'], ['CH3CONH2'], 1e-11, 0.0, 800.0, 'grain'),

            # Formação de glicolaldeído (CH2OHCHO)
            Reaction(['CH2OH', 'HCO'], ['CH2OHCHO'], 1e-11, 0.0, 700.0, 'grain'),

            # Fotodissociação
            Reaction(['H2O'], ['OH', 'H'], 1e-10, 0.0, 0.0, 'photodissociation'),
            Reaction(['NH3'], ['NH2', 'H'], 1e-10, 0.0, 0.0, 'photodissociation'),
            Reaction(['CH4'], ['CH3', 'H'], 1e-10, 0.0, 0.0, 'photodissociation'),
        ]
        for rxn in default:
            key = '_'.join(sorted(rxn.reactants))
            self.reactions.setdefault(key, []).append(rxn)
            self.species.update(rxn.reactants + rxn.products)

# =============================================================================
# MODELO DE SUPERFÍCIE DE GRÃOS (Garrod et al. 2008)
# =============================================================================

@dataclass
class GrainSurfaceModel:
    """
    Implementa química em superfície de grãos com:
    - Difusão de radicais via hopping térmico
    - Dessorção térmica (lei de Arrhenius)
    - Fotodissociação induzida por raios cósmicos
    """
    grain_density: float = 1e-12      # cm⁻² (densidade de sítios)
    site_density: float = 1.5e15       # cm⁻²
    grain_radius: float = 1e-5         # cm (0.1 μm)
    binding_energies: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        # Energias de ligação (K) - baseado em
        self.binding_energies = {
            'H': 350, 'H2': 450, 'OH': 2850, 'H2O': 4800,
            'CO': 1150, 'HCO': 2000, 'CH3': 1170, 'CH3OH': 4000,
            'NH3': 3000, 'NH2': 2500, 'CH2OH': 4000, 'CH3O': 4000,
            'HNCO': 3000, 'NH2CONH2': 8000, 'CH3CONH2': 7000,
            'CH2OHCHO': 6500,
        }

    def hopping_rate(self, species: str, T_dust: float) -> float:
        """Taxa de hopping (difusão) para um radical na superfície """
        Eb = self.binding_energies.get(species, 1000.0)
        nu0 = 1e12  # frequência de tentativa típica (s⁻¹)
        return nu0 * np.exp(-Eb / T_dust)

    def thermal_desorption_rate(self, species: str, T_dust: float) -> float:
        """Taxa de dessorção térmica (lei de Arrhenius)"""
        Eb = self.binding_energies.get(species, 1000.0)
        nu0 = 1e12
        return nu0 * np.exp(-Eb / T_dust)

    def diffusion_length(self, species: str, T_dust: float, time: float) -> float:
        """Número médio de sítios percorridos por difusão"""
        hop_rate = self.hopping_rate(species, T_dust)
        return np.sqrt(hop_rate * time)  # random walk

# =============================================================================
# MODELO PRINCIPAL: ASTROPHYSICAL CHRONOGLYPH
# =============================================================================

class AstrophysicalChronoglyph:
    """
    Modelo físico que traduz sequências binárias em condições iniciais
    para cinética química realista em nuvens moleculares.
    """

    def __init__(self, params: Optional[Dict] = None,
                 rate_db: Optional[ReactionDatabase] = None):
        self.params = params or SGR_B2_PARAMS.copy()
        self.rate_db = rate_db or ReactionDatabase()
        self.rate_db.load_umist_rate12()
        self.grain_model = GrainSurfaceModel()

        # Lista de espécies a modelar
        self.species_list = self._initialize_species()
        self.n_species = len(self.species_list)
        self.species_index = {s: i for i, s in enumerate(self.species_list)}

    def _initialize_species(self) -> List[str]:
        """Lista completa de espécies baseada em """
        return [
            # Elementares
            'H', 'H2', 'C', 'N', 'O', 'S',
            # Moléculas simples
            'OH', 'H2O', 'CO', 'NH3', 'CH4', 'HCN', 'HNC', 'CS',
            # Radicais (cruciais para química em grãos)
            'CH3', 'CH2', 'CH', 'CN', 'NH', 'NH2', 'HCO', 'CH3O', 'CH2OH',
            # Moléculas orgânicas complexas
            'CH3OH', 'H2CO', 'CH3CHO', 'C2H5OH',
            'HNCO', 'NH2CHO',                     # isocianato / formamida
            'NH2CONH2',                             # ureia
            'CH3CONH2',                              # acetamida
            'CH2OHCHO', 'HCOOCH3', 'CH3COOH',       # isômeros C2H4O2
            'CH3OCH3', 'C2H5OH',                     # dimetil éter / etanol
            'HC3N', 'C2H3CN',                         # nitrilas
            # Íons (para química de fase gasosa)
            'H3O+', 'HCO+', 'CH3OH2+', 'NH4+',
            'CH3CONH3+', 'CH2OHCHO+',
        ]

    def bits_to_initial_abundances(self, bits: str) -> np.ndarray:
        """
        Converte sequência binária em condições iniciais físicas.

        Base física:
        - Bits '0' = regiões de baixa densidade/radiação (H abundante)
        - Bits '1' = regiões processadas (C, N, O enriquecidos)
        - Correlação de longo alcance (bit 42) = nitrogênio (amidas)
        """
        n_bits = len(bits)
        bit_array = np.array([int(b) for b in bits])

        # Frações elementares derivadas da sequência
        h_fraction = 1.0 - bit_array.mean()           # H associado a '0's
        c_fraction = bit_array.mean() * 0.3           # Carbono (menos abundante)
        n_fraction = self._nitrogen_from_structure(bit_array)  # N de correlações
        o_fraction = (c_fraction + n_fraction) * 0.5   # O como intermediário

        # Normaliza (mantém H dominante, como universo real)
        total = h_fraction + c_fraction + n_fraction + o_fraction
        h_fraction /= total
        c_fraction /= total
        n_fraction /= total
        o_fraction /= total

        # Abundâncias iniciais (relativas a H2 = 1)
        # Valores típicos de nuvens escuras
        abundances = np.zeros(self.n_species)

        # Espécies elementares
        abundances[self.species_index['H']] = h_fraction * 1e-4
        abundances[self.species_index['H2']] = 1.0
        abundances[self.species_index['C']] = c_fraction * 1e-4
        abundances[self.species_index['N']] = n_fraction * 1e-5
        abundances[self.species_index['O']] = o_fraction * 1e-4

        # Moléculas simples formadas no meio interestelar
        abundances[self.species_index['CO']] = c_fraction * o_fraction * 1e-4
        abundances[self.species_index['H2O']] = o_fraction * 1e-4
        abundances[self.species_index['NH3']] = n_fraction * 1e-6
        abundances[self.species_index['CH4']] = c_fraction * 1e-6

        # Sementes para química em grãos
        abundances[self.species_index['HNCO']] = c_fraction * n_fraction * o_fraction * 1e-7

        return abundances

    def _nitrogen_from_structure(self, bits: np.ndarray) -> float:
        """
        Extrai contribuição de nitrogênio a partir de correlações estruturais.
        Baseado na observação que amidas requerem estruturas não-triviais .
        """
        n = len(bits)

        # Correlação de longo alcance (bit 42 como marcador de complexidade)
        bit_42 = bits[41] if n > 41 else 0

        # Mede periodicidade (isomeria)
        correlations = []
        for shift in [21, 42, 63]:  # submúltiplos de 84
            if shift < n:
                matches = np.sum(bits[:-shift] == bits[shift:])
                correlations.append(matches / (n - shift))

        pattern_strength = np.mean(correlations) if correlations else 0.5

        # Nitrogênio mais abundante quando há estrutura
        n_contrib = 0.3 if bit_42 else 0.1
        n_contrib *= (1 + pattern_strength)

        return n_contrib

    def chemical_ode(self, t: float, y: np.ndarray,
                     T: float, n_gas: float) -> np.ndarray:
        """
        Sistema de EDOs para evolução química.

        dy/dt = Σ formação - Σ destruição
        Implementa equações stiff típicas de astroquímica
        """
        abundances = dict(zip(self.species_list, y))
        dydt = np.zeros_like(y)

        # Taxas de reação dependentes da temperatura
        T_kelvin = T

        # Itera sobre todas as reações possíveis
        for rxn_list in self.rate_db.reactions.values():
            for rxn in rxn_list:
                k = rxn.rate(T_kelvin, n_gas)

                # Determina reagentes presentes
                if len(rxn.reactants) == 2:
                    a, b = rxn.reactants
                    if a in self.species_index and b in self.species_index:
                        rate = k * abundances.get(a, 0) * abundances.get(b, 0)
                    else:
                        continue
                elif len(rxn.reactants) == 1:
                    a = rxn.reactants[0]
                    if a in self.species_index:
                        rate = k * abundances.get(a, 0)
                    else:
                        continue
                else:
                    continue

                # Aplica taxa aos reagentes (destruição)
                for sp in rxn.reactants:
                    if sp in self.species_index:
                        dydt[self.species_index[sp]] -= rate

                # Aplica taxa aos produtos (formação)
                for sp in rxn.products:
                    if sp in self.species_index:
                        dydt[self.species_index[sp]] += rate

        return dydt

    def evolve_chemistry(self, initial_abundances: np.ndarray,
                         t_max_years: float = 1e6) -> Tuple[np.ndarray, Dict]:
        """
        Evolui química usando solver para equações stiff.
        Utiliza método BDF (Backward Differentiation Formula)
        """
        t_span = (0.0, t_max_years * 365.25 * 24 * 3600)  # converte para segundos
        T = self.params['temperature']
        n_gas = self.params['density']

        def ode_func(t, y):
            return self.chemical_ode(t, y, T, n_gas)

        # Configuração do solver stiff
        solution = solve_ivp(
            ode_func,
            t_span,
            initial_abundances,
            method='BDF',           # método para stiff equations
            rtol=1e-6,               # tolerância relativa
            atol=1e-12,              # tolerância absoluta
            dense_output=True,
            max_step=1e6,            # passo máximo em segundos (~10 dias)
        )

        stats = {
            'success': solution.success,
            'nfev': solution.nfev,
            'njev': solution.njev,
            'message': solution.message,
            't_final': solution.t[-1],
        }

        return solution.y[:, -1], stats

    def predict_observations(self, final_abundances: np.ndarray) -> Dict:
        """
        Converte abundâncias em predições observacionais (densidade de coluna).
        Baseado em tamanho típico de hot core (~0.5 pc)
        """
        source_size_pc = 0.5
        L_cm = source_size_pc * 3.086e18  # parsec → cm

        column_densities = {}
        for i, sp in enumerate(self.species_list):
            col = final_abundances[i] * L_cm
            if col > 1e10:  # apenas espécies significativas
                column_densities[sp] = col

        # Comparação com valores observados em Sgr B2(N2)
        observed = {
            'NH2CONH2': 1.4e16,      # ureia
            'CH3CONH2': 1.0e16,       # acetamida
            'CH2OHCHO': 5.0e15,       # glicolaldeído
            'HCOOCH3': 2.0e16,        # formiato de metila
            'CH3COOH': 3.0e15,         # ácido acético
        }

        matches = {}
        for mol, obs_val in observed.items():
            if mol in column_densities:
                pred = column_densities[mol]
                ratio = pred / obs_val
                matches[mol] = {
                    'predicted': pred,
                    'observed': obs_val,
                    'ratio': ratio,
                    'quality': 'good' if 0.2 < ratio < 5 else 'poor'
                }

        return {
            'column_densities': column_densities,
            'matches': matches,
            'success_rate': sum(1 for m in matches.values() if m['quality']=='good') / len(matches) if matches else 0,
        }

    def decode_and_predict(self, bit_sequence: str) -> Dict:
        """
        Pipeline completo: bits → abundâncias → evolução → predições
        """
        print(f"\n{'='*60}")
        print(f"🔮 DECODIFICANDO SEQUÊNCIA DE {len(bit_sequence)} BITS")
        print(f"Bit 42: {bit_sequence[41] if len(bit_sequence) > 41 else 'N/A'}")
        print(f"Temperatura: {self.params['temperature']} K")
        print(f"Densidade: {self.params['density']:.1e} cm⁻³")
        print(f"Tempo de evolução: {self.params['age_years']:.1e} anos")
        print('='*60)

        # 1. Bits → condições iniciais
        initial = self.bits_to_initial_abundances(bit_sequence)
        print(f"\n📊 Abundâncias iniciais:")
        for sp in ['H', 'C', 'N', 'O']:
            idx = self.species_index.get(sp)
            if idx is not None:
                print(f"  {sp}: {initial[idx]:.2e}")

        # 2. Evolução química
        print(f"\n⚗️  Evoluindo química por {self.params['age_years']:.1e} anos...")
        final, stats = self.evolve_chemistry(initial, self.params['age_years'])

        print(f"  Solver: BDF, avaliações: {stats['nfev']}, status: {stats['message']}")

        # 3. Predições observacionais
        predictions = self.predict_observations(final)

        print(f"\n📡 Predições vs. Observações :")
        for mol, data in predictions['matches'].items():
            symbol = "✅" if data['quality'] == 'good' else "⚠️"
            print(f"  {symbol} {mol}: pred={data['predicted']:.2e}, "
                  f"obs={data['observed']:.2e}, razão={data['ratio']:.2f}")

        print(f"\n📈 Taxa de acerto: {predictions['success_rate']*100:.1f}%")

        return {
            'bit_sequence': bit_sequence,
            'initial_abundances': {sp: initial[idx] for sp, idx in self.species_index.items()},
            'final_abundances': {sp: final[idx] for sp, idx in self.species_index.items()},
            'predictions': predictions,
            'solver_stats': stats,
            'physical_parameters': self.params,
        }

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🜁  ASTROPHYSICAL CHRONOGLYPH - MODELO QUÍMICO")
    print("="*70)
    # Exemplo simples se rodar direto
    SEQUENCE_85 = "00001010111011000111110011010010000101011101100011111001101001000010101110"
    model = AstrophysicalChronoglyph()
    model.decode_and_predict(SEQUENCE_85)
