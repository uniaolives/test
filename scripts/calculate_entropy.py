# scripts/calculate_entropy.py
import sys
import math
import subprocess

def calculate_shannon_entropy(data):
    if not data:
        return 0
    entropy = 0
    for x in range(256):
        p_x = float(data.count(chr(x)))/len(data)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    return entropy

def get_staged_files_content():
    try:
        files = subprocess.check_output(['git', 'diff', '--cached', '--name-only']).decode().splitlines()
        content = ""
        for f in files:
            try:
                content += open(f, 'r').read()
            except:
                pass
        return content
    except:
        return ""

if __name__ == "__main__":
    content = get_staged_files_content()
    # Scaled to represent system complexity in the cosmopsychia context
    entropy = calculate_shannon_entropy(content) * 10
    print(f"{entropy:.2f}")
