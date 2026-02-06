# scripts/calculate_entropy.py
import sys
import math
import subprocess

def calculate_shannon_entropy(data):
    if not data:
        return 0
    entropy = 0
    freq = {}
    for char in data:
        freq[char] = freq.get(char, 0) + 1

    length = len(data)
    for count in freq.values():
        p_x = float(count) / length
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    return entropy

def get_staged_content():
    try:
        files = subprocess.check_output(['git', 'diff', '--cached', '--name-only']).decode().splitlines()
        content = ""
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                    content += file.read()
            except Exception:
                pass
        return content
    except Exception:
        return ""

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--staged":
        content = get_staged_content()
    elif len(sys.argv) > 1:
        filename = sys.argv[1]
        try:
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            content = ""
    else:
        print("0.00")
        sys.exit(0)

    entropy = calculate_shannon_entropy(content) * 10
    print(f"{entropy:.2f}")
