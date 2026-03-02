#!/bin/bash
# scripts/verify_hardened_build.sh

set -e

BINARY="target/release/tiger_init_hardened"

if [ ! -f "$BINARY" ]; then
    echo "❌ Binary not found. Run 'cargo build --release -p sasc_core --bin tiger_init_hardened' first."
    exit 1
fi

echo "🔍 Verificando Tiger-51 Binary Hardening..."

# ORDEM 1: Verificar que panic = "abort" está ativo
if grep -q 'panic = "abort"' rust/Cargo.toml; then
    echo "✅ panic = \"abort\" verified in Cargo.toml"
else
    echo "❌ CRITICAL: panic = \"abort\" NOT FOUND in Cargo.toml"
    exit 1
fi

# ORDEM 3: Verificar que símbolos foram removidos (strip)
if grep -q "strip = true" rust/Cargo.toml; then
    echo "✅ strip = true verified in Cargo.toml"
else
    echo "❌ CRITICAL: strip = true NOT FOUND in Cargo.toml"
    exit 1
fi

# Verificar se APK verifier está presente (marcador de string)
if ! strings "$BINARY" | grep -q "META-INF/CERT.RSA"; then
    echo "❌ CRITICAL: APK verifier logic NOT FOUND in binary"
    exit 1
fi

# Verificar se zeroize está presente (marcador no binário)
if strings "$BINARY" | grep -q "zeroize"; then
    echo "✅ zeroize markers found"
else
    echo "❌ CRITICAL: zeroize markers NOT FOUND"
    exit 1
fi

echo "✅ Hardening orders verified for binary: $BINARY"
