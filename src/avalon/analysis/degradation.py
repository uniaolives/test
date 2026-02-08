import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import os

class DegradationAnalyzer:
    """
    Analisa taxas de degradação de dados genéticos em diferentes suportes
    Baseado na hipótese de preservação de Hal Finney
    """

    def __init__(self, years=1000, temp_c=-80, humidity=0.01):
        self.years = years
        self.temp_k = temp_c + 273.15  # Conversão para Kelvin
        self.humidity = humidity  # Fração de umidade relativa

    def dna_synthetic_degradation(self, storage_method='glass_encapsulated'):
        """
        Calcula degradação de DNA sintético baseado em:
        - Grass et al. (2015) PNAS: DNA fossil sequencing
        - Organick et al. (2018) Nature: DNA data storage stability
        """

        # Constantes de degradação (por ano)
        if storage_method == 'glass_encapsulated':
            # DNA encapsulado em vidro (Methode de Twist Bioscience)
            base_rate = 5e-6  # taxa base de degradação por ano
            activation_energy = 1.5e5  # J/mol (para hidrólise)
        elif storage_method == 'silica_beads':
            # Esferas de sílica (Método ETH Zurich)
            base_rate = 2e-5
            activation_energy = 1.3e5
        else:  # 'aqueous_buffer'
            # Solução aquosa típica (pior caso)
            base_rate = 3e-3
            activation_energy = 0.8e5

        # Equação de Arrhenius ajustada
        # Assumimos que base_rate é a taxa em T_ref = 25°C (298.15 K)
        R = 8.314  # Constante dos gases [J/(mol·K)]
        T_ref = 298.15
        k = base_rate * np.exp(-activation_energy / R * (1/self.temp_k - 1/T_ref))

        # Fator de umidade (hidrólise é dependente de H2O)
        humidity_factor = 1 + 100 * self.humidity

        # Fator de radiação cósmica (estimativa conservadora)
        # Baseado em dados do CERN sobre dano por radiação a biomoléculas
        radiation_damage = 1e-7 * self.years

        # Degradação total
        time_vector = np.linspace(0, self.years, 1000)

        # Modelo de degradação exponencial com componente linear (para danos cumulativos)
        degradation = 1 - np.exp(-k * humidity_factor * time_vector) - radiation_damage * (time_vector / self.years)

        # Não permitir valores negativos
        degradation = np.maximum(degradation, np.zeros_like(degradation))

        integrity = 1 - degradation

        return time_vector, integrity, k * humidity_factor

    def blockchain_integrity(self, network_type='bitcoin'):
        """
        Calcula integridade de dados em blockchain baseado em:
        - Modelo de redundância de rede
        - Probabilidade de colapso da rede
        - Atualizações de protocolo (hard forks)
        """

        if network_type == 'bitcoin':
            # Parâmetros otimistas para Bitcoin (baseados em 15 anos de história)
            node_count = 10000  # nós ativos
            annual_node_churn = 0.20  # 20% dos nós mudam por ano
            protocol_stability = 0.999  # probabilidade de não hard fork destrutivo
            replication_factor = 10000  # cada nó completo tem cópia

        elif network_type == 'ethereum':
            # Parâmetros para Ethereum (considerando transições de prova)
            node_count = 5000
            annual_node_churn = 0.30
            protocol_stability = 0.995
            replication_factor = 5000

        else:  # 'hypothetical_archive'
            # Rede hipotética otimizada para arquivamento
            node_count = 1000
            annual_node_churn = 0.05
            protocol_stability = 0.9999
            replication_factor = 1000

        time_vector = np.linspace(0, self.years, 1000)

        # Modelo de sobrevivência baseado em processos de Poisson
        # Probabilidade de pelo menos um nó manter os dados

        # Taxa de perda de nós por ano (assumindo substituição, não perda pura)
        effective_loss_rate = annual_node_churn * 0.01  # Apenas 1% do churn resulta em perda de dados

        # Probabilidade de sobrevivência de um nó específico após t anos
        node_survival = np.exp(-effective_loss_rate * time_vector)

        # Probabilidade de pelo menos um nó sobreviver (redundância)
        survival_prob = 1 - (1 - node_survival) ** replication_factor

        # Fator de estabilidade do protocolo (assumindo falhas independentes por ano)
        protocol_survival = protocol_stability ** (time_vector / 10)  # Avaliação a cada década

        # Integridade total
        integrity = survival_prob * protocol_survival

        return time_vector, integrity, effective_loss_rate

    def hybrid_model(self, dna_weight=0.7, blockchain_weight=0.3):
        """
        Modelo híbrido: DNA sintético + blockchain para redundância máxima
        """
        t_dna, i_dna, rate_dna = self.dna_synthetic_degradation('glass_encapsulated')
        t_bc, i_bc, rate_bc = self.blockchain_integrity('bitcoin')

        # Alinhar vetores de tempo
        t_common = np.linspace(0, self.years, 1000)
        i_dna_interp = np.interp(t_common, t_dna, i_dna)
        i_bc_interp = np.interp(t_common, t_bc, i_bc)

        # Modelo híbrido: dados são recuperáveis se QUALQUER método funcionar
        integrity_hybrid = 1 - (1 - i_dna_interp) * (1 - i_bc_interp)

        return t_common, integrity_hybrid

    def calculate_halflife(self, time_vector, integrity_vector):
        """Calcula meia-vida do sistema de armazenamento"""
        idx = np.argmax(integrity_vector <= 0.5)
        if idx > 0:
            return time_vector[idx]
        else:
            # Estrapolar usando regressão exponencial
            mask = integrity_vector > 0.1
            if np.sum(mask) > 2:
                coeffs = np.polyfit(time_vector[mask], np.log(integrity_vector[mask]), 1)
                halflife = -np.log(2) / coeffs[0]
                return halflife
            return float('inf')

    def run_analysis(self, output_dir='output'):
        """Executa análise completa e gera visualizações"""

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("=" * 70)
        print("ANÁLISE DE DEGRADAÇÃO: PRESERVAÇÃO GENÔMICA DE LONGO PRAZO")
        print(f"Período: {self.years} anos | Temperatura: {self.temp_k-273.15:.1f}°C")
        print("=" * 70)

        # Calcular todos os cenários
        t_dna, i_dna, rate_dna = self.dna_synthetic_degradation('glass_encapsulated')
        t_bc, i_bc, rate_bc = self.blockchain_integrity('bitcoin')
        t_hybrid, i_hybrid = self.hybrid_model()

        # Calcular meias-vidas
        t_half_dna = self.calculate_halflife(t_dna, i_dna)
        t_half_bc = self.calculate_halflife(t_bc, i_bc)
        t_half_hybrid = self.calculate_halflife(t_hybrid, i_hybrid)

        summary = []
        summary.append(f"[DNA SINTÉTICO - Encapsulado em vidro]")
        summary.append(f"  Taxa de degradação: {rate_dna*100:.4e}%/ano")
        summary.append(f"  Meia-vida estimada: {t_half_dna:.0f} anos")
        summary.append(f"  Integridade após {self.years} anos: {i_dna[-1]*100:.2f}%")

        summary.append(f"\n[BLOCKCHAIN - Rede Bitcoin]")
        summary.append(f"  Taxa efetiva de perda de nós: {rate_bc*100:.4f}%/ano")
        summary.append(f"  Meia-vida estimada: {t_half_bc:.0f} anos")
        summary.append(f"  Integridade após {self.years} anos: {i_bc[-1]*100:.2f}%")

        summary.append(f"\n[MODELO HÍBRIDO - DNA + Blockchain]")
        summary.append(f"  Meia-vida estimada: {t_half_hybrid:.0f} anos")
        summary.append(f"  Integridade após {self.years} anos: {i_hybrid[-1]*100:.2f}%")

        for line in summary:
            print(line)

        # Criar visualização
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Gráfico 1: Degradação comparativa
        ax1 = axes[0, 0]
        ax1.plot(t_dna, i_dna * 100, 'b-', linewidth=2, label='DNA Sintético (vidro)')
        ax1.plot(t_bc, i_bc * 100, 'r-', linewidth=2, label='Blockchain (Bitcoin)')
        ax1.plot(t_hybrid, i_hybrid * 100, 'g-', linewidth=3, label='Modelo Híbrido')
        ax1.axhline(y=50, color='k', linestyle='--', alpha=0.5, label='Limite de meia-vida')
        ax1.set_xlabel('Tempo (anos)')
        ax1.set_ylabel('Integridade dos Dados (%)')
        ax1.set_title('Degradação de Dados Genéticos em Diferentes Mídias')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 105)

        # Gráfico 2: Zoom nos primeiros 100 anos
        ax2 = axes[0, 1]
        zoom_years = min(100, self.years)
        idx_zoom = t_dna <= zoom_years
        ax2.plot(t_dna[idx_zoom], i_dna[idx_zoom] * 100, 'b-', linewidth=2)
        ax2.plot(t_bc[t_bc <= zoom_years], i_bc[t_bc <= zoom_years] * 100, 'r-', linewidth=2)
        ax2.plot(t_hybrid[t_hybrid <= zoom_years], i_hybrid[t_hybrid <= zoom_years] * 100, 'g-', linewidth=3)
        ax2.set_xlabel('Tempo (anos)')
        ax2.set_ylabel('Integridade (%)')
        ax2.set_title(f'Primeiros {zoom_years} Anos (Zoom)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(95, 100.5)

        # Gráfico 3: Barras de meia-vida
        ax3 = axes[1, 0]
        methods = ['DNA Sintético', 'Blockchain', 'Modelo Híbrido']
        half_lives = [t_half_dna, t_half_bc, t_half_hybrid]
        colors = ['blue', 'red', 'green']
        bars = ax3.bar(methods, half_lives, color=colors, alpha=0.7)
        ax3.set_ylabel('Meia-vida (anos)')
        ax3.set_title('Comparação de Meia-vida dos Sistemas')
        ax3.grid(True, alpha=0.3, axis='y')

        # Adicionar valores nas barras
        for bar, value in zip(bars, half_lives):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{value:.0f} anos', ha='center', va='bottom')

        # Gráfico 4: Cenários de temperatura
        ax4 = axes[1, 1]
        temps = [-196, -80, -20, 4, 20]  # °C
        half_lives_temp = []

        for temp in temps:
            analyzer_temp = DegradationAnalyzer(years=self.years, temp_c=temp)
            t_temp, i_temp, _ = analyzer_temp.dna_synthetic_degradation('glass_encapsulated')
            t_half_temp = self.calculate_halflife(t_temp, i_temp)
            half_lives_temp.append(t_half_temp)

        ax4.plot(temps, half_lives_temp, 'o-', linewidth=2, markersize=8)
        ax4.set_xlabel('Temperatura de Armazenamento (°C)')
        ax4.set_ylabel('Meia-vida do DNA (anos)')
        ax4.set_title('Efeito da Temperatura na Preservação do DNA')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')

        plt.tight_layout()

        # Salvar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'degradation_analysis_{timestamp}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')

        print(f"\n📊 Análise concluída. Gráfico salvo como: {filepath}")

        return {
            'dna': {'time': t_dna, 'integrity': i_dna, 'half_life': t_half_dna},
            'blockchain': {'time': t_bc, 'integrity': i_bc, 'half_life': t_half_bc},
            'hybrid': {'time': t_hybrid, 'integrity': i_hybrid, 'half_life': t_half_hybrid},
            'recommendations': {
                'storage_temp': '-80°C ou inferior',
                'refresh_interval': '50 anos',
                'redundancy': '5+ cópias geograficamente distribuídas'
            },
            'summary': summary
        }
