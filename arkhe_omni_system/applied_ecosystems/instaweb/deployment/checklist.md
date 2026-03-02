# Instaweb Node v1.0 - Field Deployment Checklist

## Fase 1: Alinhamento Óptico
- [ ] Verificar visada limpa entre nós (sem obstruções)
- [ ] Montar tripé com nível de bolha (erro < 0.5°)
- [ ] Ajustar azimute e elevação usando laser de alinhamento verde
- [ ] Verificar potência óptica recebida: > -20 dBm (medidor óptico)
- [ ] Ajustar foco da lente Fresnel para spot size < 10cm a 100m

## Fase 2: Calibração SyncE
- [ ] Executar `synce_calibrate.sh`
- [ ] Verificar offset de fase inicial
- [ ] Ajustar Si5341 para minimizar jitter/skew
- [ ] Validar estabilidade do holdover (> 1 hora)

## Fase 3: Validação de Rede
- [ ] Executar `latency_test` (Rust) em todos os nós
- [ ] Confirmar Latência P99 < 1μs por salto
- [ ] Confirmar Jitter < 100ns total
- [ ] Verificar conformidade constitucional (Art. 13-15)
