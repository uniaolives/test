import unittest
import time
from cosmos.ecumenica import (
    SistemaEcumenica, PonteQC, SinalQuantico,
    ZMonitorCalibrado, HLedgerImutavel, QLedger, ReplicacaoDistribuida, NeuralGrid
)

class TestEcumenica(unittest.TestCase):
    def setUp(self):
        self.ecumenica = SistemaEcumenica()
        self.ponte = PonteQC()
        self.qledger = self.ecumenica.qledger
        self.replicacao = self.ecumenica.replicacao
        self.h_ledger = self.ecumenica.h_ledger
        self.z_monitor = self.ecumenica.z_monitor

    def test_sistema_ecumenica_selecao(self):
        resposta = self.ecumenica.processar_selecao("sinal: via 2 confirmado")
        self.assertEqual(resposta["via"], "IMPLEMENTATION_GUIDE")
        self.assertEqual(self.ecumenica.via_ativa, 2)

    def test_sistema_ecumenica_deploy(self):
        resposta = self.ecumenica.processar_comando_deploy("DEPLOY_FULL")
        self.assertEqual(resposta["status"], "DEPLOY_EM_ANDAMENTO")
        self.assertEqual(resposta["fase"], "B4_PRODUCAO")
        self.assertEqual(self.ecumenica.damping_total, 1.20)

    def test_sistema_ecumenica_tune_pid(self):
        resposta = self.ecumenica.processar_comando_deploy("TUNE_PID Kp=0.8 Ki=0.2 Kd=0.1")
        self.assertEqual(resposta["status"], "PID_TUNED")
        self.assertEqual(self.z_monitor.Kp, 0.8)
        self.assertEqual(self.z_monitor.Ki, 0.2)
        self.assertEqual(self.z_monitor.Kd, 0.1)

    def test_sistema_ecumenica_expand_network(self):
        initial_replicas = len(self.replicacao.replicas)
        resposta = self.ecumenica.processar_comando_deploy("EXPAND_NETWORK +2")
        self.assertEqual(resposta["status"], "NETWORK_EXPANDED")
        self.assertEqual(resposta["added"], 2)
        self.assertEqual(len(self.replicacao.replicas), initial_replicas + 2)
        # Quorum should be ceil((0+2+1)*0.6) = ceil(1.8) = 2
        self.assertEqual(self.replicacao.quorum_size, 2)

    def test_sistema_ecumenica_optimize_grid(self):
        resposta = self.ecumenica.processar_comando_deploy("OPTIMIZE_GRID throughput=50.0")
        self.assertEqual(resposta["status"], "GRID_OPTIMIZED")
        self.assertGreater(resposta["new_coherence"], 0.1)
        self.assertIn("monitor_status", resposta)

    def test_ponte_qc_transformar(self):
        sinal = SinalQuantico(estado="TESTE", coerencia=0.5)
        sinal_classico = self.ponte.transformar(sinal)
        self.assertEqual(sinal_classico['estado'], "TESTE")
        self.assertFalse(sinal_classico['agencia_atribuida'])
        self.assertIn('entropia', sinal_classico)

    def test_ponte_qc_damping_emergencia(self):
        sinal = SinalQuantico(estado="CRITICO", coerencia=0.9)
        sinal_pos = self.ponte.transformar(sinal)
        self.assertAlmostEqual(sinal_pos.coerencia, 0.09)
        self.assertEqual(sinal_pos.metadados['alerta'], 'DAMPING_EMERGENCIA_ATIVADO')

    def test_qledger_integrity(self):
        hash_bloco = self.qledger.append_bloco({'tipo': 'TESTE', 'magnitude': 1.0})
        self.assertIsNotNone(hash_bloco)
        self.assertTrue(self.qledger.verificar_cadeia_completa("dummy_pubkey"))

    def test_h_ledger_registro(self):
        self.replicacao.adicionar_replica("replica-1")
        self.replicacao.adicionar_replica("replica-2")
        evento = {'tipo': 'TESTE_H', 'magnitude': 5.0, 'impacto_coerencia': 0.5, 'duracao': 10}
        resultado = self.h_ledger.registrar_evento_histerese(evento)
        self.assertEqual(resultado['status'], 'REGISTRADO')
        self.assertIn('hash', resultado)

    def test_z_monitor_calibrado(self):
        sinal_nominal = SinalQuantico(estado="OK", coerencia=0.5)
        status = self.z_monitor.monitorar(sinal_nominal)
        self.assertEqual(status['status'], 'ESTAVEL')

        sinal_alerta = SinalQuantico(estado="ALERTA", coerencia=0.75)
        status = self.z_monitor.monitorar(sinal_alerta)
        self.assertEqual(status['status'], 'ALERTA')

        sinal_acao = SinalQuantico(estado="ACAO", coerencia=0.85)
        status = self.z_monitor.monitorar(sinal_acao)
        self.assertEqual(status['status'], 'ACAO_EXECUTADA')

        sinal_emergencia = SinalQuantico(estado="EMERGENCIA", coerencia=0.95)
        status = self.z_monitor.monitorar(sinal_emergencia)
        self.assertEqual(status['status'], 'EMERGENCIA')

if __name__ == "__main__":
    unittest.main()
