# bridge_integration.py
# Ponte: Python Frontend → Rust Core

import subprocess
import json
import os
import sys

# Garantir que o diretório atual está no path para importar pyfrontend
sys.path.append(os.getcwd())

from pyfrontend.constitutional_lowering import CCIRLoweringLayer
from pyfrontend.energy_analysis import EnergyAnalyzer
from pyfrontend.graph_capture import GraphCaptureLayer

class CognitiveBridgeClient:
    """
    Cliente Python que orquestra a transpilação e invoca o compilador Rust
    """

    def __init__(self, rust_compiler_path="./target/release/cognitive"):
        self.compiler_path = rust_compiler_path
        self.capture_layer = GraphCaptureLayer()
        self.energy_analyzer = EnergyAnalyzer()
        self.lowering_layer = CCIRLoweringLayer()

    def compile_model(self, python_code_path: str) -> dict:
        """
        Pipeline Completo:
        Python → Grafo → CCIR → Arquivo Temporário → Rust Compiler → WASM
        """

        print(f"Bridge [BRIDGE] Iniciando Ponte Cognitiva para: {python_code_path}")

        # 1. Ler código fonte Python
        if not os.path.exists(python_code_path):
            return {"status": "error", "message": f"Arquivo não encontrado: {python_code_path}"}

        with open(python_code_path, 'r') as f:
            source_code = f.read()

        # 2. Capturar Grafo Computacional
        print("   [1/5] Capturando Grafo Computacional...")
        try:
            graph = self.capture_layer.capture(source_code)
        except Exception as e:
            return {"status": "error", "message": f"Falha na captura do grafo: {str(e)}"}

        # 3. Análise Energética (A \"Física\" do Código)
        print("   [2/5] Calculando Perfil Energético...")
        energy_profile = self.energy_analyzer.analyze_graph(graph)
        print(f"       Energia Estimada: {energy_profile.total_energy:.6f} Joules")

        # 4. Lowering para CCIR (Conversão para Dialeto Constitucional)
        print("   [3/5] Gerando CCIR...")
        ccir_code = self.lowering_layer.lower(
            graph,
            energy_profile,
            constraints=None # Usar padrões da constituição
        )

        # 5. Salvar CCIR temporário
        temp_ccir_path = "/tmp/module.ccir"
        with open(temp_ccir_path, 'w') as f:
            f.write(ccir_code)

        # 6. Invocar Compilador Rust (Fase 0)
        print("   [4/5] Invocando Compilador Constitucional (Rust Core)...")

        try:
            # Chamada ao binário Rust criado na Fase 0
            result = subprocess.run(
                [self.compiler_path, "compile", temp_ccir_path, "--language", "ccir"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                print("   [5/5] Compilação Bem-Sucedida!")
                # O mock do Rust retorna o path do WASM no stdout
                lines = result.stdout.strip().split('\n')
                wasm_output = lines[-1] if lines else "N/A"

                return {
                    "status": "success",
                    "energy_consumed": energy_profile.total_energy,
                    "output_wasm": wasm_output,
                    "constitutional_log": result.stdout
                }
            else:
                # O compilador Rust rejeitou (Violou constituição)
                print("   [!] REJEITADO PELO CORE CONSTITUCIONAL")
                return {
                    "status": "rejected",
                    "reason": "Constitutional Violation",
                    "compiler_output": result.stderr
                }

        except FileNotFoundError:
            print(f"   ⚠️  Compilador Rust não encontrado em: {self.compiler_path}. Fallback: Simulação.")
            return {
                "status": "simulated_success",
                "message": "Rust core não encontrado, mas CCIR gerado com sucesso.",
                "ccir_preview": ccir_code[:200] + "..."
            }

# Uso Exemplo
if __name__ == "__main__":
    client = CognitiveBridgeClient()

    file_to_compile = sys.argv[1] if len(sys.argv) > 1 else "examples/simple_transpilation.py"

    # Tentar compilar um modelo Python
    result = client.compile_model(file_to_compile)

    print("\n📦 RESULTADO:")
    print(json.dumps(result, indent=2))
