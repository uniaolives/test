import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metalanguage.latent_catalysis import LatentCatalysisModel
from metalanguage.astrophysical_chronoglyph_final import SGR_B2_PARAMS

def run_catalysis_sweep(m_eff_values):
    """
    Executes chemical simulations for a range of m_eff values
    and returns final acetamide abundances.
    """
    results = []
    print(f"Running catalysis sweep for {len(m_eff_values)} points...")

    for m in m_eff_values:
        model = LatentCatalysisModel(params=SGR_B2_PARAMS, m_eff=m)

        # Initial abundances typical for dark clouds / Sgr B2
        initial = np.zeros(model.n_species)
        idx = model.species_index
        initial[idx['H']] = 1e-4
        initial[idx['H2']] = 1.0
        initial[idx['C']] = 1e-4
        initial[idx['N']] = 1e-5
        initial[idx['O']] = 1e-4
        initial[idx['CO']] = 1e-4
        initial[idx['H2O']] = 1e-4
        initial[idx['NH3']] = 1e-6
        initial[idx['CH4']] = 1e-6
        initial[idx['HNCO']] = 1e-7
        initial[idx['CH3']] = 1e-8
        initial[idx['CONH2']] = 1e-8

        # Evolve chemistry for 1e6 years
        final, stats = model.evolve_chemistry(initial, t_max_years=1e6)
        acetamide_abundance = final[idx['CH3CONH2']]
        results.append(acetamide_abundance)
        print(f"  m_eff={m:.2f} -> Acetamide={acetamide_abundance:.2e}")

    return results

def test_catalysis_experiment():
    m_eff_range = np.linspace(0, 2, 5) # Smaller range for test speed
    abundances = run_catalysis_sweep(m_eff_range)

    # Verification: Acetamide should increase with m_eff
    assert abundances[-1] > abundances[0], "Latent catalysis failed to increase abundance"

    # Visualization
    plt.figure(figsize=(8,5))
    plt.plot(m_eff_range, abundances, 'o-', color='purple', label='Simulated')
    plt.xlabel('Effective Mass Field $m_{\\mathrm{eff}}$')
    plt.ylabel('Final Acetamide Abundance (rel. H$_2$)')
    plt.title('Latent Catalysis: Impact of m_eff on Acetamide Synthesis')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('catalysis_sweep_analysis.png')
    print("✅ Catalysis sweep plot saved to catalysis_sweep_analysis.png")

if __name__ == "__main__":
    try:
        test_catalysis_experiment()
        print("\nLatent Catalysis Sweep Successful! 🧪🌀")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
