# API Reference

## Arkhe Core (porta 8080)
- `GET /status` – retorna estado do nó (id, coherence, satoshi, handovers)
- `POST /handover` – realiza handover para outro nó (body: {to, payload})
- `GET /anticipate` – retorna predição de coerência futura

## GLP Server (porta 5000)
- `POST /encode` – codifica ativação em meta‑neurônios (body: {activation})
- `POST /steer` – aplica direção de conceito (body: {meta, direction, strength})
