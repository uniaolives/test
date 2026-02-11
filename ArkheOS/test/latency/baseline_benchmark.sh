# test/latency/baseline_benchmark.sh
#!/bin/bash
# Mede latência atual do sistema via ZeroMQ

# 1. Instala sockperf
apt-get install -y sockperf

# 2. Node q1 (servidor)
docker exec -d arkhe-node-1 sockperf server -i 0.0.0.0 -p 11111

# 3. Node q2 (cliente)
docker exec arkhe-node-2 sockperf ping-pong -i arkhe-node-1 -p 11111 \
    --tcp -t 60 -m 64 --full-log baseline_latency.log

# 4. Analisa resultados
python test/latency/analyze_baseline.py baseline_latency.log
