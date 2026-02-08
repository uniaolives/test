import pytest
import numpy as np
import os
from src.avalon.analysis.degradation import DegradationAnalyzer

def test_degradation_analyzer_initialization():
    analyzer = DegradationAnalyzer(years=100, temp_c=-20)
    assert analyzer.years == 100
    assert analyzer.temp_k == -20 + 273.15

def test_dna_degradation_logic():
    analyzer = DegradationAnalyzer(years=1000, temp_c=20)
    t, integrity, rate = analyzer.dna_synthetic_degradation('aqueous_buffer')

    # At 20C in water, DNA should degrade significantly over 1000 years
    assert integrity[-1] < 0.5
    assert rate > 0

def test_blockchain_integrity_logic():
    analyzer = DegradationAnalyzer(years=100)
    t, integrity, rate = analyzer.blockchain_integrity('bitcoin')

    # Bitcoin should be fairly stable over 100 years in the model
    assert integrity[-1] > 0.8
    assert rate > 0

def test_hybrid_model():
    analyzer = DegradationAnalyzer(years=500)
    t, integrity = analyzer.hybrid_model()

    t_dna, i_dna, _ = analyzer.dna_synthetic_degradation('glass_encapsulated')
    t_bc, i_bc, _ = analyzer.blockchain_integrity('bitcoin')

    # Hybrid should be at least as good as the best individual method
    assert integrity[-1] >= i_dna[-1]
    assert integrity[-1] >= i_bc[-1]

def test_halflife_calculation():
    analyzer = DegradationAnalyzer()
    time = np.linspace(0, 1000, 1000)
    # Artificial exponential decay: e^(-kt) = 0.5 => t = ln(2)/k
    k = 0.001
    integrity = np.exp(-k * time)
    expected_halflife = np.log(2) / k

    calc_halflife = analyzer.calculate_halflife(time, integrity)
    assert pytest.approx(calc_halflife, rel=1e-2) == expected_halflife

def test_run_analysis(tmp_path):
    output_dir = tmp_path / "analysis_output"
    analyzer = DegradationAnalyzer(years=50)
    results = analyzer.run_analysis(output_dir=str(output_dir))

    assert os.path.exists(output_dir)
    assert len(results['summary']) > 0
    # Check if a plot was saved (it might have a timestamp)
    plots = list(output_dir.glob("*.png"))
    assert len(plots) > 0
