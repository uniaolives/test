# node_client/agent_node.py
import asyncio
import json
import os
from typing import Dict, Any
import nats
from nats.js.api import StreamConfig
import psutil
try:
    import GPUtil
except ImportError:
    GPUtil = None

class AgentNode:
    """Cliente de nó distribuído para SASC Network"""

    def __init__(self):
        self.node_id = os.environ.get('NODE_ID', 'unknown')
        self.node_type = os.environ.get('NODE_TYPE', 'generic')  # mac_mini, pc, laptop
        self.vps_gateway = os.environ.get('VPS_GATEWAY')
        self.capabilities = self._detect_capabilities()
        self.current_load = 0.0

    def _detect_capabilities(self) -> Dict[str, Any]:
        """Detecta hardware e capacidades do nó"""
        capabilities = {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'cpu': {
                'cores': psutil.cpu_count(),
                'freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
                'architecture': os.uname().machine if hasattr(os, 'uname') else 'unknown'
            },
            'memory': {
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'available_gb': psutil.virtual_memory().available / (1024**3)
            },
            'gpu': self._detect_gpu(),
            'storage': {
                'total_gb': psutil.disk_usage('/').total / (1024**3),
                'free_gb': psutil.disk_usage('/').free / (1024**3)
            },
            'specializations': self._detect_specializations()
        }
        return capabilities

    def _detect_gpu(self) -> Dict[str, Any]:
        """Detecta GPUs disponíveis"""
        if GPUtil:
            try:
                gpus = GPUtil.getGPUs()
                return [{
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_gb': gpu.memoryTotal / 1024,
                    'load': gpu.load
                } for gpu in gpus]
            except:
                pass

        # macOS Metal ou CPU-only
        if self.node_type == 'mac_mini':
            return [{'type': 'apple_silicon', 'name': 'M1/M2/M3', 'metal': True}]
        return []

    def _detect_specializations(self) -> list:
        """Detecta especializações baseadas no hardware"""
        specs = []

        if self.node_type == 'mac_mini':
            specs.extend(['knowledge_base', 'local_llm', 'file_storage'])
        elif self.node_type == 'pc' and self._detect_gpu():
            specs.extend(['compute_heavy', 'training', 'simulation'])
        elif self.node_type == 'laptop':
            specs.extend(['mobile_interface', 'edge_processing', 'cache'])

        return specs

    async def connect_to_brain(self):
        """Conecta ao VPS Brain via NATS"""
        nc = await nats.connect(f"nats://{self.vps_gateway}:4222")
        js = nc.jetstream()

        # Registrar no stream de nodes
        await js.publish(
            "nodes.register",
            json.dumps(self.capabilities).encode()
        )

        # Subscrever em tarefas
        sub = await js.subscribe(
            f"tasks.{self.node_id}",
            durable=self.node_id
        )

        print(f"🟢 Node {self.node_id} conectado ao Brain")
        print(f"   Capacidades: {self.capabilities['specializations']}")

        async for msg in sub:
            await self._process_task(msg)

    async def _process_task(self, msg):
        """Processa tarefa recebida do Brain"""
        task = json.loads(msg.data.decode())

        print(f"📥 Tarefa recebida: {task['type']} (ID: {task['id']})")

        # Executar tarefa baseada no tipo
        if task['type'] == 'inference':
            result = await self._run_inference(task['data'])
        elif task['type'] == 'file_process':
            result = await self._process_file(task['data'])
        elif task['type'] == 'security_scan':
            result = await self._security_scan(task['data'])
        else:
            result = {'error': 'unknown_task_type'}

        # Reportar resultado
        await msg.ack()
        await self._report_result(task['id'], result)

    async def _run_inference(self, data: Dict) -> Dict:
        """Executa inferência local (Ollama/Llama.cpp)"""
        import subprocess

        model = data.get('model', 'llama2')
        prompt = data.get('prompt', '')

        # Chamar Ollama local
        try:
            result = subprocess.run(
                ['ollama', 'run', model, prompt],
                capture_output=True,
                text=True,
                timeout=120
            )
            output = result.stdout
        except Exception as e:
            output = str(e)

        return {
            'output': output,
            'model': model,
            'node': self.node_id
        }

    async def _process_file(self, data: Dict) -> Dict:
        """Processa arquivo localmente"""
        # Implementação específica por tipo de arquivo
        filepath = data.get('path')
        operation = data.get('operation', 'index')

        # Indexar para RAG, transformar, etc.
        return {
            'file': filepath,
            'operation': operation,
            'status': 'completed',
            'node': self.node_id
        }

    async def _security_scan(self, data: Dict) -> Dict:
        """Executa scan de segurança local"""
        # Integração com security-agent
        return {
            'scan_type': data.get('type'),
            'findings': [],
            'node': self.node_id
        }

    async def _report_result(self, task_id: str, result: Dict):
        """Reporta resultado de volta ao Brain"""
        nc = await nats.connect(f"nats://{self.vps_gateway}:4222")
        js = nc.jetstream()

        await js.publish(
            "tasks.results",
            json.dumps({
                'task_id': task_id,
                'node_id': self.node_id,
                'result': result,
                'timestamp': asyncio.get_event_loop().time()
            }).encode()
        )

        print(f"📤 Resultado reportado: {task_id}")

# Execução
if __name__ == "__main__":
    node = AgentNode()
    try:
        asyncio.run(node.connect_to_brain())
    except KeyboardInterrupt:
        pass
