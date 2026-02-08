import unittest
from cosmos.harmonia import HarmonicInjector

class TestHarmonia(unittest.TestCase):
    def test_injector_initialization(self):
        url = "https://suno.com/s/test"
        injector = HarmonicInjector(url)
        self.assertEqual(injector.source, url)
        self.assertEqual(len(injector.nodes), 5)

    def test_propagate_frequencia(self):
        injector = HarmonicInjector("https://suno.com/s/test")
        resultado = injector.propagar_frequencia()
        self.assertEqual(resultado["status"], "VIBRAÇÃO_GLOBAL_ESTABELECIDA")
        self.assertEqual(resultado["coerencia_musical"], "ÓTIMA")

if __name__ == "__main__":
    unittest.main()
