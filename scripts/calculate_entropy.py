#!/usr/bin/env python3
"""
Calcula a entropia de Shannon do código para o filtro Rosehip.
"""

import math
import os
import sys
from pathlib import Path

def calculate_file_entropy(filepath):
    """Calcula entropia de Shannon de um arquivo."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        if not content:
            return 0.0

        # Frequência de caracteres
        freq = {}
        for char in content:
            freq[char] = freq.get(char, 0) + 1

        # Entropia de Shannon
        entropy = 0.0
        total_chars = len(content)

        for count in freq.values():
            probability = count / total_chars
            if probability > 0:
                entropy -= probability * math.log2(probability)

        return entropy

    except Exception as e:
        return 0.0

def detect_patterns(filepath):
    """Detecta padrões humanos no código."""
    patterns = {
        'has_todos': 0,
        'has_comments': 0,
        'has_tests': 0,
        'has_logging': 0,
    }

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        for line in lines:
            line_lower = line.lower()

            if any(todo in line_lower for todo in ['todo:', 'fixme:', 'optimize:', 'hack:']):
                patterns['has_todos'] = 1

            if line.strip().startswith('#') or line.strip().startswith('//'):
                patterns['has_comments'] = 1

            if any(test in line_lower for test in ['def test_', 'test(', 'it(', 'describe(']):
                patterns['has_tests'] = 1

            if any(log in line_lower for log in ['log.', 'console.', 'print', 'debug']):
                patterns['has_logging'] = 1

    except:
        pass

    return patterns

def main():
    if len(sys.argv) < 2:
        # Modo padrão: calcular para todo o repo
        repo_path = Path.cwd()
        total_entropy = 0.0
        file_count = 0

        for ext in ['.py', '.js', '.ts', '.rs', '.c', '.cpp', '.java']:
            for filepath in repo_path.rglob(f'*{ext}'):
                if '.git' in str(filepath):
                    continue

                entropy = calculate_file_entropy(filepath)
                patterns = detect_patterns(filepath)

                # Ajustar entropia baseado em padrões humanos
                human_factor = sum(patterns.values()) / len(patterns)
                adjusted_entropy = entropy * (1.0 - human_factor * 0.2)

                total_entropy += adjusted_entropy
                file_count += 1

        if file_count > 0:
            avg_entropy = total_entropy / file_count
            print(f"{avg_entropy:.2f}")
        else:
            print("0.0")
    else:
        # Calcular para arquivo específico
        filepath = sys.argv[1]
        entropy = calculate_file_entropy(filepath)
        print(f"{entropy:.2f}")

if __name__ == "__main__":
    main()
