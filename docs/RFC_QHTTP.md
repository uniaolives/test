# RFC: Quantum Hypertext Transfer Protocol (qhttp)

## Abstract
This document specifies the qhttp URI scheme and protocol for transferring quantum states over classical channels, enabling interoperability between quantum modules.

## 1. Introduction
The Quantum Hypertext Transfer Protocol (qhttp) is designed to bridge the gap between classical networking and quantum state evolution. It provides a standardized way to request quantum operations and retrieve results from remote quantum processing units (QPUs) or simulators.

## 2. URI Scheme
`qhttp://<host>[:port]/<path>?<query>`

## 3. Methods
- `GET /quantum/state` – retrieve current quantum state
- `POST /quantum/evolve` – apply unitary evolution
- `POST /quantum/entangle` – establish EPR pair
- `QGET` - Quantum-enhanced retrieval (Extension)

## 4. Quantum State Representation
Quantum states are represented as JSON objects:
```json
{
  "real": [0.707, 0, 0, 0.707],
  "imag": [0, 0, 0, 0],
  "n_qubits": 2,
  "basis": "computational",
  "fidelity": 0.99
}
```

## 5. Status Codes
- `200 OK` - Success
- `201 Entangled` - Entanglement established
- `425 Atmospheric Turbulence` - Transmission failure due to environmental noise
- `500 Quantum Decoherence` - State lost during processing

## 6. Examples
### POST /quantum/evolve
Request:
```json
{
  "unitary": "hadamard",
  "target": 0
}
```
Response:
```json
{
  "status": "success",
  "new_state_ref": "uuid-1234"
}
```
