#!/bin/bash
# monitor.sh
# Real-time metrics monitoring for ASI-Omega Mesh
# Usage: bash monitor.sh <ip1> <ip2> ...

IPS=("$@")
if [ ${#IPS[@]} -eq 0 ]; then
  echo "Usage: bash monitor.sh <ip1> <ip2> ..."
  exit 1
fi

while true; do
  clear
  echo "=== ASI-Ω REAL-TIME METRICS ==="
  for ip in "${IPS[@]}"; do
    result=$(curl -s --max-time 1 http://$ip:3000/metrics)
    if [ $? -eq 0 ]; then
      c=$(echo $result | jq -r '.c_global_estimate')
      n=$(echo $result | jq -r '.neighbors_count')
      echo "$ip : C=$c N=$n"
    else
      echo "$ip : OFFLINE"
    fi
  done
  sleep 5
done
