import sys
import os

# Add metalanguage to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metalanguage.heptapod_interpreter import LinguagemComoHeptapod, HeptapodBytecodeInterpreter

def test_heptapod_interpreter():
    interpreter = HeptapodBytecodeInterpreter()
    test_bits = "000010101110110001111100110100100001010111"

    # Test Lisp
    lisp_results = interpreter.interpret(test_bits, "lisp")
    expected_lisp = ["Macro recursiva", "Eval de lista", "Apply parcial", "Quasiquote"]
    assert all(item in lisp_results for item in expected_lisp)
    print("✅ Lisp interpretations correct")

    # Test Haskell
    haskell_results = interpreter.interpret(test_bits, "haskell")
    expected_haskell = ["Função lazy infinita", "Monad IO", "Composição (>=>)", "Type constructor"]
    assert all(item in haskell_results for item in expected_haskell)
    print("✅ Haskell interpretations correct")

    # Test Prolog
    prolog_results = interpreter.interpret(test_bits, "prolog")
    expected_prolog = ["Predicado com cut", "Fail/Backtrack", "Unificação (=)", "Assert/retract"]
    assert all(item in prolog_results for item in expected_prolog)
    print("✅ Prolog interpretations correct")

def test_language_model():
    haskell = LinguagemComoHeptapod("Haskell", "funcional")
    prolog = LinguagemComoHeptapod("Prolog", "logico")

    fusion = haskell.merge(prolog)
    assert fusion.nome == "Haskell+Prolog"
    assert "funcional" in fusion.fragmento_semantico
    assert "logico" in fusion.fragmento_semantico
    assert fusion.fragmento_semantico["funcional"] == "tempo_como_demand"
    assert fusion.fragmento_semantico["logico"] == "causalidade_reversivel"
    print("✅ Language merge correct")

if __name__ == "__main__":
    try:
        test_heptapod_interpreter()
        test_language_model()
        print("\nAll Heptapod tests passed! 🌀")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        sys.exit(1)
