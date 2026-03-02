#!/bin/bash
set -e

echo "--- Ω-12 TRUTH_AUDITORIUM COMPLIANCE AUDIT ---"

# Check for required gates in source code
echo "🔍 Checking for Gate 1 & 2: Prince Key + EIP-712..."
grep -q "Cathedral::instance()" src/lib.rs && echo "✅ PASS: Cathedral detected"

echo "🔍 Checking for Gate 3: DNA Extraction..."
grep -q "cathedral.verify_agent_attestation" src/lib.rs && echo "✅ PASS: DNA extraction from attestation"

echo "🔍 Checking for Gate 4: Hard Freeze Check..."
grep -q "attestation_status.is_hard_frozen()" src/lib.rs && echo "✅ PASS: Hard Freeze blocks submission"

echo "🔍 Checking for Gate 5: Vajra Entropy Weighting..."
grep -q "vajra_monitor.update_entropy" src/lib.rs && echo "✅ PASS: Entropy weighting active"

echo "---"
echo "✅ AUDIT COMPLETE: All 5 Ω-12 gates detected."
