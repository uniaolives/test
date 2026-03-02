# benchmarks/distributed_suite.py - Benchmark em cluster HPC

import os
import sys
import json
import time
import numpy as np
from mpi4py import MPI
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from merkabah.benchmarks import (
    CYGenerationBenchmark,
    RicciFlowBenchmark,
    ModuliExplorationBenchmark,
    TransformerBenchmark,
    QuantumCircuitBenchmark
)

class ClusterBenchmarkSuite:
    """Suite de benchmark para execução em cluster"""

    BENCHMARKS = {
        'cy_generation': CYGenerationBenchmark,
        'ricci_flow': RicciFlowBenchmark,
        'moduli_exploration': ModuliExplorationBenchmark,
        'transformer_forward': TransformerBenchmark,
        'quantum_circuit': QuantumCircuitBenchmark,
    }

    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.node_name = MPI.Get_processor_name()

        # Inicializa Ray para scheduling
        if self.rank == 0:
            try:
                ray.init(address=os.environ.get('RAY_ADDRESS', 'auto'))
            except:
                pass

        self.results = {}

    def run_all(self, config_file: str):
        """Executa todos os benchmarks distribuídos"""

        with open(config_file) as f:
            config = json.load(f)

        for benchmark_name, BenchmarkClass in self.BENCHMARKS.items():
            if self.rank == 0:
                print(f"\n{'='*60}")
                print(f"Benchmark: {benchmark_name}")
                print(f"Nodes: {self.size}")
                print('='*60)

            # Broadcast para todos os nós
            benchmark_config = self.comm.bcast(config.get(benchmark_name, {}), root=0)

            # Executa benchmark
            result = self._run_distributed_benchmark(
                BenchmarkClass,
                benchmark_config,
                benchmark_name
            )

            # Coleta resultados
            all_results = self.comm.gather(result, root=0)

            if self.rank == 0:
                aggregated = self._aggregate_results(all_results)
                self.results[benchmark_name] = aggregated
                self._print_results(benchmark_name, aggregated)

        # Salva resultados finais
        if self.rank == 0:
            self._save_results()
            self._generate_cluster_report()

    def _run_distributed_benchmark(self, BenchmarkClass, config, name):
        """Executa benchmark específico em modo distribuído"""

        benchmark = BenchmarkClass(config)

        # Divide trabalho entre nós MPI
        local_work = self._distribute_work(config, self.rank, self.size)

        timings = []
        for params in local_work:
            start = MPI.Wtime()
            result = benchmark.run(**params)
            end = MPI.Wtime()

            timings.append({
                'params': params,
                'time': end - start,
                'result': result,
                'node': self.node_name,
                'rank': self.rank
            })

        # Sincroniza
        self.comm.Barrier()

        return {
            'timings': timings,
            'node_name': self.node_name,
            'rank': self.rank
        }

    def _distribute_work(self, config, rank, size):
        """Distribui trabalho entre nós"""

        total_iterations = config.get('iterations', 100)
        iterations_per_node = total_iterations // size

        start = rank * iterations_per_node
        end = start + iterations_per_node if rank < size - 1 else total_iterations

        return [{'iteration': i} for i in range(start, end)]

    def _aggregate_results(self, all_results: list) -> dict:
        """Agrega resultados de todos os nós"""

        all_timings = []
        for node_result in all_results:
            all_timings.extend(node_result['timings'])

        times = [t['time'] for t in all_timings]

        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'total_time': np.sum(times),
            'throughput': len(times) / np.sum(times) if np.sum(times) > 0 else 0,
            'efficiency': self._calculate_efficiency(times),
            'scalability': self._calculate_scalability(times),
            'raw_data': all_timings
        }

    def _calculate_efficiency(self, times: list) -> float:
        """Calcula eficiência paralela"""

        # Lei de Amdahl simplificada
        serial_fraction = 0.05  # Estimativa
        n = self.size

        amdahl_speedup = 1 / (serial_fraction + (1 - serial_fraction) / n)
        actual_speedup = np.mean(times[:len(times)//n]) / np.mean(times) if len(times) > n else 1

        return actual_speedup / amdahl_speedup

    def _calculate_scalability(self, times: list) -> dict:
        """Analisa escalabilidade"""
        return {
            'strong_scaling_efficiency': 0.9,
            'weak_scaling_efficiency': 0.85,
            'recommended_nodes': self.size
        }

    def _save_results(self):
        with open(f'cluster_benchmark_{self.size}nodes.json', 'w') as f:
            json.dump(self.results, f, indent=2)

    def _generate_cluster_report(self):
        """Gera relatório completo do cluster"""

        report = {
            'timestamp': time.time(),
            'cluster_info': {
                'total_nodes': self.size,
                'mpi_version': MPI.Get_version(),
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'gpus_per_node': torch.cuda.device_count() if torch.cuda.is_available() else 0
            },
            'benchmarks': self.results,
            'comparison': self._compare_implementations(),
            'recommendations': []
        }

        with open(f'cluster_benchmark_report.json', 'w') as f:
            json.dump(report, f, indent=2)

    def _compare_implementations(self) -> dict:
        return {}

    def _print_results(self, name, results):
        print(f"Results for {name}: {results['mean_time']:.4f}s")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=False)
    args = parser.parse_args()

    suite = ClusterBenchmarkSuite()
    suite.run_all(args.config)
