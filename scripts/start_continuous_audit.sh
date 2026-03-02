#!/bash/bin
# scripts/start_continuous_audit.sh

# 1. Iniciar loop de auditoria
echo "🚀 Starting Continuous Audit Loop..."
cargo run --release -- \
    audit start \
    --contract 0x42A5D99c \
    --quantum-seed "$QUANTUM_SEED" \
    --interval 30 \
    --jni-env "$ANDROID_JNI_ENV" \
    --android-context "$ANDROID_CONTEXT" \
    --log-level info \
    --output json \
    --daemon

# 2. Iniciar dashboard
echo "📊 Starting Audit Dashboard..."
cd dashboard
# npm run build
# npm run preview -- --port 8080 --host 0.0.0.0 &

# 3. Gerar relatório inicial de conformidade
echo "📋 Generating Initial Compliance Report..."
python3 scripts/generate_compliance_report.py 0x42A5D99c builds/app-release.apk

# 4. Monitorar
echo "👁️ Monitoring audit logs..."
# tail -f logs/audit_$(date +%Y%m%d).log | grep -E "(CRITICAL|ERROR|WARN)"
