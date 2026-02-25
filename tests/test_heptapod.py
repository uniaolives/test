import sys
import os
import unittest

# Add the root directory to sys.path to allow importing from metalanguage
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metalanguage.heptapod_model import (
    Semagram, GraphSentence, HeptapodProcessor,
    MolecularFragment, MolecularGraph, ProgrammingLanguageAtlas
)

class TestHeptapodModel(unittest.TestCase):
    def setUp(self):
        self.processor = HeptapodProcessor()
        self.h01 = Semagram("H01", "human", 3, 1.0)
        self.h02 = Semagram("H02", "arrive", 2, 0.9)
        self.processor.learn_semagram(self.h01)
        self.processor.learn_semagram(self.h02)

    def test_basic_sentence(self):
        sentence = GraphSentence("S1")
        sentence.add_semagram(self.h01)
        sentence.add_semagram(self.h02)
        sentence.connect("H01", "H02", "agent")

        self.assertEqual(len(sentence.semagrams), 2)
        self.assertEqual(len(sentence.edges), 1)

        coherence = sentence.calculate_coherence()
        self.assertGreater(coherence, 0)

    def test_merge_sentences(self):
        s1 = GraphSentence("S1")
        s1.add_semagram(self.h01)

        s2 = GraphSentence("S2")
        s2.add_semagram(self.h02)

        merged = s1.merge(s2)
        self.assertEqual(len(merged.semagrams), 2)
        self.assertIn("H01", merged.semagrams)
        self.assertIn("H02", merged.semagrams)

    def test_85bit_decoding(self):
        # 5 bits for N=3 (00011) + 3 bits for adjacency matrix (111) + padding
        bit_string = "00011" + "111" + "0" * (85 - 8)
        sentence = HeptapodProcessor.from_85bit_string(bit_string)

        self.assertEqual(len(sentence.semagrams), 3)
        # For N=3, matrix is (0,1), (0,2), (1,2) -> 3 edges if all bits are 1
        self.assertEqual(len(sentence.edges), 3)

    def test_molecular_mapping(self):
        m1 = MolecularFragment("OH", "hydroxyl", 1, formula="OH")
        mol = MolecularGraph("Ethanol")
        mol.add_semagram(m1)
        self.assertEqual(mol.get_molecular_weight(), 14.0)

    def test_language_atlas(self):
        atlas = ProgrammingLanguageAtlas()
        lisp_sentence = atlas.get_language_sentence("Lisp")
        self.assertIsNotNone(lisp_sentence)
        self.assertIn("Lisp", lisp_sentence.semagrams)
        self.assertIn("Recursão", lisp_sentence.semagrams)

    def test_demo_run(self):
        # This just ensures the demo function doesn't crash
        from metalanguage.heptapod_model import demo_heptapod_communication
        # Redirect stdout to avoid cluttering test output
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            demo_heptapod_communication()
        self.assertIn("HEPTAPOD B", f.getvalue())

if __name__ == "__main__":
    unittest.main()

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
