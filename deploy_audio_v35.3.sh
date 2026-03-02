#!/bin/bash
# deploy_audio_v35.3.sh
set -euo pipefail

echo "🔊🎶 Deploy Audio Engine v35.3-Ω"

# 1. Compilar
echo "🔨 Compilando Audio Engine..."
cargo build --release -p cge-audio-engine

echo "✅ Audio Engine operacional!"
