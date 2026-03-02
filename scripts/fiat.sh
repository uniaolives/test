#!/bin/bash
# scripts/fiat.sh
# Bash wrapper for the fiat CLI

echo "📜 FIAT PROTOCOL INTERFACE"
python3 $(dirname $0)/fiat.py "$@"
