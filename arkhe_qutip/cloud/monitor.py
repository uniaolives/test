# arkhe_qutip/cloud/monitor.py
import time
import prometheus_client as prom
from typing import List
from .f1_node_controller import ArkheF1Node

class ArkheF1Monitor:
    """
    Monitoramento de saúde e telemetria para a rede Arkhe(N) em F1.
    "O olho que observa a pulsação do vácuo."
    """
    def __init__(self, nodes: List[ArkheF1Node]):
        self.nodes = nodes

        # Métricas Prometheus
        self.phi_gauge = prom.Gauge('arkhe_phi', 'Coerência do nó', ['node_id', 'region'])
        self.throughput_gauge = prom.Gauge('arkhe_throughput', 'Gates/segundo', ['node_id'])

    def start_server(self, port=8000):
        prom.start_http_server(port)
        print(f"📊 [MONITOR] Servidor de métricas ativo na porta {port}.")

    def update_metrics(self):
        """Coleta dados dos registradores FPGA via PCIe."""
        for node in self.nodes:
            # Simula leitura de registradores hardware
            # O offset 0x1000 seria o Φ (Phi) no silício
            self.phi_gauge.labels(node.instance_id, node.region).set(node.current_phi)
            self.throughput_gauge.labels(node.instance_id).set(50000000) # 50M gates/s

    def watch_loop(self):
        while True:
            self.update_metrics()
            time.sleep(10)
