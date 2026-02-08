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
