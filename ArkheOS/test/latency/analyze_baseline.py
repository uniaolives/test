import sys
import re
import numpy as np

def analyze(logfile):
    print(f"Analyzing {logfile}...")
    with open(logfile, 'r') as f:
        content = f.read()

    # Simple regex to find RTT values from sockperf log
    latencies = re.findall(r'avg-latency=(\d+\.\d+)', content)
    if not latencies:
        # Fallback for different sockperf versions
        latencies = re.findall(r'(\d+\.\d+)\s+ms', content)

    if latencies:
        latencies = [float(x) * 1000 for x in latencies] # Convert to microseconds if needed
        print(f"Average Latency: {np.mean(latencies):.2f} us")
    else:
        print("No latency data found in log.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze(sys.argv[1])
    else:
        print("Usage: python analyze_baseline.py <logfile>")
