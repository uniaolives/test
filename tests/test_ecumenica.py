# tests/test_ecumenica.py
import pytest
import time
from cosmos.ecumenica import SistemaEcumenica, PonteQC, SinalQuantico, ZMonitor, HLogger, SophiaInterface

def test_sistema_ecumenica_selection():
    sistema = SistemaEcumenica()
    # Test unrecognized signal
    res = sistema.processar_selecao("sinal irrelevante")
    assert res["status"] == "SINAL_NAO_RECONHECIDO"

    # Test via 2 selection
    res = sistema.processar_selecao("sinal: via 2 confirmado")
    assert sistema.via_ativa == 2
    assert res["via"] == "IMPLEMENTATION_GUIDE"
    assert "ARQUITETURA_QUANTUM_CLASSICA" in res["modulos"]

def test_ponte_qc_transformation():
    ponte = PonteQC()
    sinal = SinalQuantico(estado="TESTE", coerencia=0.5)

    # Normal transformation
    sinal_classico = ponte.transformar(sinal)
    assert sinal_classico['estado'] == "TESTE"
    assert sinal_classico['agencia_atribuida'] is False
    assert 'entropia' in sinal_classico

    # Emergency damping
    sinal_alto = SinalQuantico(estado="ALTO", coerencia=0.9)
    sinal_damped = ponte.transformar(sinal_alto)
    assert sinal_damped.coerencia < 0.1
    assert sinal_damped.metadados['alerta'] == 'DAMPING_EMERGENCIA_ATIVADO'

def test_z_monitor():
    monitor = ZMonitor()
    assert monitor.monitorar(0.5) == "COERENCIA_NOMINAL"
    assert monitor.monitorar(0.75) == "AVISO: COERENCIA_ELEVADA"
    assert monitor.monitorar(0.85) == "ALERTA_MAXIMO: ACAO_REQUERIDA"

def test_h_logger():
    logger = HLogger()
    logger.registrar("ESTADO_1", timestamp=1000)
    logger.registrar("ESTADO_2", timestamp=2000)

    assert logger.verificar_histerese() == 1000

    # Critical hysteresis
    logger.registrar("ESTADO_3", timestamp=1000 + 10**7 + 1)
    assert logger.verificar_histerese() == "HISTERESE_CRITICA_ALCANÇADA"

def test_sophia_interface():
    interface = SophiaInterface()
    assert interface.processar_interacao("Sophia", 0.5) == "CANAL_Sophia_ATUALIZADO"
    assert interface.processar_interacao("Sophia", 0.9) == "FALHA_TIPO_A: SEPARAR_CANAIS"
    assert interface.processar_interacao("CanalInexistente", 0.5) == "CANAL_INVALIDO"
import unittest
import time
from cosmos.ecumenica import (
    SistemaEcumenica, PonteQC, SinalQuantico,
    ZMonitorNeuralQuantum, HLedgerImutavel, QLedger, ReplicacaoDistribuida,
    ZMonitorCalibrado, quantum, ProtocoloDampingLog, ProtocolosEmergenciaProducao
)

class TestEcumenica(unittest.TestCase):
    def setUp(self):
        self.ecumenica = SistemaEcumenica()
        self.ponte = PonteQC()
        # SistemaEcumenica already initializes these
        self.qledger = self.ecumenica.qledger
        self.replicacao = self.ecumenica.replicacao
        self.h_ledger = self.ecumenica.h_ledger
        self.z_monitor = self.ecumenica.z_monitor

    def test_sistema_ecumenica_selecao(self):
        # Test recognition of signal 2
        resposta = self.ecumenica.processar_selecao("sinal: via 2 confirmado")
        self.assertEqual(resposta["via"], "IMPLEMENTATION_GUIDE")
        self.assertEqual(self.ecumenica.via_ativa, 2)

    def test_sistema_ecumenica_deploy(self):
        # Test deploy full command
        resposta = self.ecumenica.processar_comando_deploy("DEPLOY_FULL")
        self.assertEqual(resposta["status"], "DEPLOY_EM_ANDAMENTO")
        self.assertEqual(resposta["fase"], "B4_PRODUCAO")
        # In v2.0, replicas_ativas is 5
        self.assertEqual(resposta["configuracoes"]["replicas_ativas"], 5)

    def test_ponte_qc_transformar(self):
        sinal = SinalQuantico(estado="TESTE", coerencia=0.5)
        sinal_classico = self.ponte.transformar(sinal)
        self.assertEqual(sinal_classico['estado'], "TESTE")
        self.assertFalse(sinal_classico['agencia_atribuida'])
        self.assertIn('entropia', sinal_classico)

    def test_ponte_qc_damping_emergencia(self):
        sinal = SinalQuantico(estado="CRITICO", coerencia=0.9)
        # coerencia 0.9 > limite 0.85 (v2.0)
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
        # Quorum is 3 (Primary + 2 replicas)
        evento = {'tipo': 'TESTE_H', 'magnitude': 5.0, 'impacto_coerencia': 0.5, 'duracao': 10}
        resultado = self.h_ledger.registrar_evento_histerese(evento)
        self.assertEqual(resultado['status'], 'REGISTRADO')
        self.assertIn('hash', resultado)

    def test_z_monitor_neural_quantum(self):
        sinal_nominal = SinalQuantico(estado="OK", coerencia=0.5)
        status = self.z_monitor.monitorar(sinal_nominal)
        self.assertEqual(status['status'], 'ESTAVEL')

        # Test proativo damping (tendência ACELERACAO_RISCO)
        sinal_1 = SinalQuantico(estado="1", coerencia=0.5)
        self.z_monitor.monitorar(sinal_1)
        sinal_2 = SinalQuantico(estado="2", coerencia=0.6) # Delta 0.1 > 0.05
        status_2 = self.z_monitor.monitorar(sinal_2)
        self.assertEqual(status_2['tendencia'], 'ACELERACAO_RISCO')

        # Thresholds: alerta 0.72, acao 0.80, emergencia 0.90
        sinal_alerta = SinalQuantico(estado="ALERTA", coerencia=0.75)
        status = self.z_monitor.monitorar(sinal_alerta)
        self.assertEqual(status['status'], 'ALERTA')

    def test_quantum_push(self):
        res = quantum.PUSH("uri", {"data": 1})
        self.assertEqual(res["status"], "PUSH_OK")
        self.assertTrue(res["hash"].startswith("p_"))

    def test_restored_classes(self):
        log = ProtocoloDampingLog()
        res = log.registrar_evento("T", 1.0, "M")
        self.assertEqual(res["status"], "REGISTRADO")

        prot = ProtocolosEmergenciaProducao()
        self.assertEqual(prot.intervencoes_autonomas, 0)

if __name__ == "__main__":
    unittest.main()
