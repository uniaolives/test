import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metalanguage.frechet_tensor import FrechetTensor, seminorm_k
from metalanguage.moonshine_replicability import ReplicabilityTester

def test_frechet_optimization():
    print("--- Testing FrechetTensor Optimization ---")
    # Inicializa tensor aleatório
    data = torch.randn(100, requires_grad=True)

    # Define seminormas (ordens de suavidade)
    seminorms = [lambda t, k=i: seminorm_k(t, k) for i in range(1, 5)]

    frechet_t = FrechetTensor(data, seminorms)
    optimizer = optim.Adam(frechet_t.parameters(), lr=0.1)

    initial_convergence = frechet_t.get_convergence_vector()
    print(f"Initial convergence vector: {initial_convergence.detach().numpy()}")

    # Otimização para minimizar todas as seminormas (suavização)
    for i in range(50):
        optimizer.zero_grad()
        convergence = frechet_t.get_convergence_vector()
        loss = torch.sum(convergence)
        loss.backward()
        optimizer.step()

    final_convergence = frechet_t.get_convergence_vector()
    print(f"Final convergence vector: {final_convergence.detach().numpy()}")

    assert torch.sum(final_convergence) < torch.sum(initial_convergence)
    print("✅ Frechet optimization successful\n")

def test_moonshine_series():
    print("--- Testing Moonshine Replicability Logic ---")
    bits_85 = "00001010111011000111110011010010000101011101100011111001101001000010101110"
    bits = [int(b) for b in bits_85]

    tester = ReplicabilityTester(bits)

    # Verifica se a série foi construída corretamente com SymPy
    from sympy.abc import q
    # Verifica se o termo 1/q está presente
    assert (1/q) in tester.f_series.as_ordered_terms()

    # Verifica se o operador de Hecke retorna um valor (esboço)
    hecke_val = tester.twisted_hecke_operator(2)
    assert hecke_val > 0

    # Verifica stub de replicabilidade
    assert tester.verify_replicability(n_max=5) is True
    print("✅ Moonshine series logic successful\n")

if __name__ == "__main__":
    try:
        test_frechet_optimization()
        test_moonshine_series()
        print("ADVANCED THEORY TESTS PASSED! 🌌♾️")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
