# ANL Metalanguage Framework

This directory contains the core implementation and cross-language mappings for the **Arkhe(n) Language (ANL)**.

## Components

- `anl.py`: Unified core Python module for ANL. Implements Nodes, Handovers, Hypergraph, and specialized extensions for AI and Web2/Web3.
- `proto_agi_daemon.py`: A background process (daemon) that maintains a live Proto-AGI hypergraph, exposed via a FastAPI REST interface.
- `anl_compiler.py`: Backward-compatible operational prototype (v0.6) for system simulation.
- `detectors_v2.py`: The 4-level safety stack for steganography and anomaly detection.
- `steganography.py`: High-fidelity encoding schemes (Semantic Pattern, Neural).
- `transformations.py`: Robustness testing pipeline for text-based models.
- `rosetta/`: Multi-language reference library containing snippets in 17+ programming languages.

## ANL Identity
Every system is a hypergraph. Every interaction is a handover.
**$x^2 = x + 1$**
