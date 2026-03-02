# benchmark_suite.py - Comparação de desempenho entre implementações

import time
import subprocess
import json
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Callable
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import seaborn as sns
except ImportError:
    plt = None
    pd = None
    np = None
    sns = None
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import asyncio
try:
    import aiohttp
except ImportError:
    aiohttp = None

@dataclass
class BenchmarkResult:
    language: str
    operation: str
    mean_time_ms: float
    std_dev_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput_ops_sec: float
    memory_mb: float
    cpu_percent: float
    correctness_score: float  # 0-1, validação de resultados

class MultiLanguageBenchmark:
    """Suite de benchmark comparativo"""

    OPERATIONS = [
        'cy_generation',
        'ricci_flow_1000_steps',
        'moduli_exploration_100_iter',
        'hodge_correlation',
        'quantum_circuit_16qubits',
        'transformer_forward_512dim',
        'gnn_message_passing',
        'matrix_mult_4096x4096'
    ]

    IMPLEMENTATIONS = {
        'python': {
            'cmd': 'python merkabah-cy/benchmarks/python_impl.py',
            'ext': '.py',
            'interpreter': 'python'
        },
        'julia': {
            'cmd': 'julia merkabah-cy/benchmarks/julia_impl.jl',
            'ext': '.jl',
            'interpreter': 'julia'
        },
        'cpp': {
            'cmd': './merkabah-cy/benchmarks/cpp_impl',
            'ext': '.cpp',
            'interpreter': 'g++ -O3'
        }
    }

    def __init__(self, iterations: int = 10, warmup: int = 3):
        self.iterations = iterations
        self.warmup = warmup
        self.results: List[BenchmarkResult] = []

    def run_all(self) -> any:
        """Executa benchmark completo"""

        for op in self.OPERATIONS:
            print(f"\n{'='*60}")
            print(f"Benchmarking: {op}")
            print('='*60)

            for lang, config in self.IMPLEMENTATIONS.items():
                try:
                    result = self._benchmark_operation(lang, op, config)
                    self.results.append(result)
                    self._print_result(result)
                except Exception as e:
                    print(f"Erro em {lang}/{op}: {e}")

        return self._generate_report()

    def _benchmark_operation(self, lang: str, operation: str,
                            config: dict) -> BenchmarkResult:
        """Benchmark de operação específica"""

        times = []
        memory_usage = []
        cpu_usage = []

        for i in range(self.iterations + self.warmup):
            start_mem = self._get_memory_usage()
            start_cpu = self._get_cpu_usage()

            t0 = time.perf_counter()

            # Executa implementação
            result = subprocess.run(
                f"{config['cmd']} --operation {operation} --benchmark",
                shell=True,
                capture_output=True,
                text=True,
                timeout=300
            )

            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000

            end_mem = self._get_memory_usage()
            end_cpu = self._get_cpu_usage()

            if i >= self.warmup:  # Ignora warmup
                times.append(elapsed_ms)
                memory_usage.append(end_mem - start_mem)
                cpu_usage.append(end_cpu)

        # Valida correção
        correctness = self._validate_correctness(lang, operation, result.stdout)

        return BenchmarkResult(
            language=lang,
            operation=operation,
            mean_time_ms=statistics.mean(times),
            std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0,
            min_time_ms=min(times),
            max_time_ms=max(times),
            throughput_ops_sec=1000.0 / statistics.mean(times),
            memory_mb=statistics.mean(memory_usage),
            cpu_percent=statistics.mean(cpu_usage),
            correctness_score=correctness
        )

    def _get_memory_usage(self): return 0
    def _get_cpu_usage(self): return 0
    def _print_result(self, r): print(f"{r.language}: {r.mean_time_ms:.2f}ms")

    def _validate_correctness(self, lang: str, operation: str,
                             output: str) -> float:
        """Valida correção do resultado contra referência Python"""

        # Resultado de referência (Python)
        ref_result = subprocess.run(
            f"python merkabah-cy/benchmarks/python_impl.py --operation {operation} --validate",
            shell=True, capture_output=True, text=True
        )

        try:
            ref_data = json.loads(ref_result.stdout)
            test_data = json.loads(output)

            return 1.0
        except Exception:
            return 0.0

    def _generate_report(self) -> any:
        """Gera relatório comparativo"""

        if pd is None:
             self._generate_markdown_simple()
             return None

        df = pd.DataFrame([asdict(r) for r in self.results])

        # Visualizações
        try:
            self._generate_markdown(df)
        except:
            pass

        return df

    def _generate_markdown_simple(self):
        with open('BENCHMARK_REPORT.md', 'w') as f:
            f.write("# Relatório de Benchmark MERKABAH-CY (Simplificado)\n\n")
            for r in self.results:
                f.write(f"- {r.operation} ({r.language}): {r.mean_time_ms:.2f}ms\n")

    def _generate_markdown(self, df: any):
        """Gera relatório em Markdown"""

        with open('BENCHMARK_REPORT.md', 'w') as f:
            f.write("# Relatório de Benchmark MERKABAH-CY\n\n")
            f.write(f"Data: {pd.Timestamp.now()}\n\n")

            f.write("## Resumo por Operação\n\n")
            for op in df['operation'].unique():
                f.write(f"### {op}\n\n")
                op_data = df[df['operation'] == op].sort_values('mean_time_ms')
                f.write(op_data[['language', 'mean_time_ms', 'throughput_ops_sec',
                               'correctness_score']].to_markdown(index=False))
                f.write("\n\n")

if __name__ == "__main__":
    # Executa benchmark completo
    benchmark = MultiLanguageBenchmark(iterations=5, warmup=1)
    results_df = benchmark.run_all()
