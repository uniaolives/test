import pytest
import math

# Importa o módulo binário compilado do C++
try:
    import arkhe_core
except ImportError:
    pytest.fail("Módulo arkhe_core não encontrado. O Pybind11 falhou na compilação ou vinculação.")

def test_arkhe_core_binding_exists():
    """Verifica se a classe foi exportada corretamente para o Python."""
    assert hasattr(arkhe_core, "KleinBottlehole"), "Classe KleinBottlehole ausente no binding!"

def test_monodromy_bridge():
    """Testa se a lógica C++ retorna corretamente via Python."""
    topology = arkhe_core.KleinBottlehole(1.616e-35)

    # Fase causal (fechada)
    assert topology.check_monodromy_iteration(0) is False
    assert topology.check_monodromy_iteration(6) is False

    # Fase retrocausal (aberta)
    assert topology.check_monodromy_iteration(3) is True

def test_quantum_interest_bridge():
    """Testa se os cálculos de ponto flutuante sobrevivem à fronteira C++/Python."""
    topology = arkhe_core.KleinBottlehole(1.616e-35)

    dt = 1e-25
    energy = 1e15
    interest = topology.calculate_quantum_interest(dt, energy)

    assert isinstance(interest, float), "O juro quântico deve retornar como um float do Python."
    assert interest > 0.0, "O pedágio termodinâmico deve ser estritamente positivo."
    assert not math.isinf(interest), "O cálculo explodiu para o infinito na passagem FFI."
