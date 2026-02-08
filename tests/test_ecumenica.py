import unittest
import time
from cosmos.ecumenica import SistemaEcumenica, PonteQC, SinalQuantico, ZMonitor, HLogger

class TestEcumenica(unittest.TestCase):
    def setUp(self):
        self.ecumenica = SistemaEcumenica()
        self.ponte = PonteQC()
        self.z_monitor = ZMonitor()
        self.h_logger = HLogger()

    def test_sistema_ecumenica_selecao(self):
        # Test recognition of signal 2
        resposta = self.ecumenica.processar_selecao("sinal: via 2 confirmado")
        self.assertEqual(resposta["via"], "IMPLEMENTATION_GUIDE")
        self.assertEqual(self.ecumenica.via_ativa, 2)

        # Test recognition of "implementation"
        resposta = self.ecumenica.processar_selecao("Starting implementation guide")
        self.assertEqual(resposta["via"], "IMPLEMENTATION_GUIDE")

        # Test unrecognized signal
        resposta = self.ecumenica.processar_selecao("sinal aleatorio")
        self.assertEqual(resposta["status"], "SINAL_NAO_RECONHECIDO")

    def test_ponte_qc_transformar(self):
        sinal = SinalQuantico(estado="TESTE", coerencia=0.5)
        sinal_classico = self.ponte.transformar(sinal)
        self.assertEqual(sinal_classico['estado'], "TESTE")
        self.assertFalse(sinal_classico['agencia_atribuida'])
        self.assertIn('entropia', sinal_classico)

    def test_ponte_qc_damping_emergencia(self):
        sinal = SinalQuantico(estado="CRITICO", coerencia=0.9)
        # coerencia 0.9 > limite 0.8
        sinal_pos = self.ponte.transformar(sinal)
        self.assertAlmostEqual(sinal_pos.coerencia, 0.09)
        self.assertEqual(sinal_pos.metadados['alerta'], 'DAMPING_EMERGENCIA_ATIVADO')

    def test_z_monitor(self):
        sinal_nominal = SinalQuantico(estado="OK", coerencia=0.5)
        status, acao = self.z_monitor.monitorar(sinal_nominal)
        self.assertEqual(status, "COERENCIA_NOMINAL")
        self.assertFalse(acao)

        sinal_alerta = SinalQuantico(estado="ALERTA", coerencia=0.75)
        status, acao = self.z_monitor.monitorar(sinal_alerta)
        self.assertEqual(status, "ALERTA_EMITIDO")
        self.assertFalse(acao)

        sinal_acao = SinalQuantico(estado="ACAO", coerencia=0.85)
        status, acao = self.z_monitor.monitorar(sinal_acao)
        self.assertEqual(status, "ACAO_REQUERIDA")
        self.assertTrue(acao)

    def test_h_logger_arquivamento(self):
        status = self.h_logger.registrar("ESTADO_1", 1000)
        self.assertEqual(status, "REGISTRADO")
        self.assertEqual(len(self.h_logger.registros), 1)

        # Trigger archival
        status = self.h_logger.registrar("ESTADO_2", 10**7 + 1)
        self.assertEqual(status, "ESTADO_ARQUIVADO")
        self.assertEqual(len(self.h_logger.registros), 0)

if __name__ == "__main__":
    unittest.main()
