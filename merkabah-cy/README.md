# MERKABAH-CY Unified Infrastructure

This repository contains the unified infrastructure for the MERKABAH-CY framework.

## Structure

- `src/python/merkabah/`: Core Python modules, including `qhttp://` protocol and Quantum Messaging.
- `src/formal/`: Formal verification modules in Coq and Lean 4.
- `benchmarks/`: Multi-language performance comparison suite.
- `data/`: Dataset processing for Kreuzer-Skarke CY varieties.
- `.github/workflows/`: CI/CD pipelines.

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Rust (for the engine)
- Coq (for formal proofs)
- Lean 4 (for algorithm verification)

### Running the System

```bash
docker-compose up -d
```

### Running Benchmarks

```bash
python benchmarks/suite.py
```

## Critical Point: h¹·¹ = 491

The system is designed to monitor and explore the critical point at $h^{1,1} = 491$, where complexity and stability reach a delicate balance.
