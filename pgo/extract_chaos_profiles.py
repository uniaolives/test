"""
Extract profiling data from chaos test logs
Identifies hot paths in failure recovery scenarios for PGO.
"""

import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ProfileEntry:
    function_name: str
    call_count: int
    total_time_ms: float
    avg_time_ms: float
    critical_path: bool

class ChaosTestProfiler:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.profiles: Dict[str, ProfileEntry] = {}

    def parse_logs(self):
        function_calls = defaultdict(lambda: {'count': 0, 'time': 0.0})
        recovery_functions = set()

        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    match = re.search(r'\[(\d+)\] (\w+) executed in ([\d.]+)ms', line)
                    if match:
                        _, func_name, exec_time = match.groups()
                        function_calls[func_name]['count'] += 1
                        function_calls[func_name]['time'] += float(exec_time)

                    if 'RECOVERY' in line or 'RECONSTRUCTION' in line:
                        recovery_match = re.search(r'(\w+)\(', line)
                        if recovery_match:
                            recovery_functions.add(recovery_match.group(1))
        except FileNotFoundError:
            print(f"Log file {self.log_file} not found.")
            return

        for func_name, data in function_calls.items():
            self.profiles[func_name] = ProfileEntry(
                function_name=func_name,
                call_count=data['count'],
                total_time_ms=data['time'],
                avg_time_ms=data['time'] / data['count'] if data['count'] > 0 else 0,
                critical_path=func_name in recovery_functions
            )

    def generate_pgo_hints(self) -> Dict[str, str]:
        hints = {}
        for func_name, profile in self.profiles.items():
            if profile.call_count > 1000:
                hints[func_name] = 'inline'
            elif profile.critical_path:
                hints[func_name] = 'optimize_speed'
            else:
                hints[func_name] = 'default'
        return hints

if __name__ == "__main__":
    profiler = ChaosTestProfiler('chaos_test_execution.log')
    profiler.parse_logs()
    hints = profiler.generate_pgo_hints()
    with open('pgo_optimization_hints.json', 'w') as f:
        json.dump(hints, f, indent=2)
    print("✓ PGO hints generated.")
