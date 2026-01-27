
# TIM-SASC: Titanium VM & Sovereign Autonomous System Core

Welcome to the unified repository for **TIM** (Titanium Virtual Machine) and **SASC** (Sovereign Autonomous System Core). This project combines a low-level high-performance C-based virtual machine with a complex Rust-based autonomous governance and security system.

---

## 🏛️ SASC - Sovereign Autonomous System Core (v31.2-Ω)

SASC is a distributed autonomous system designed for high-coherence governance, cybersecurity, and geopolitical stability.

### Key Components:
- **Imperium Continuous Mode**: Enables autonomous governance with a 7.83Hz heartbeat synchronized to the Schumann resonance. Maintains system stability (Φ > 0.72) and executes scheduled geopolitical milestones.
- **Aletheia Metadata Engine**: Multidimensional reality verification and deepfake detection based on geometric and physical invariants.
- **Ma'at Framework**: High-density substrate navigation using holomorphic domains and Dirac fluid dynamics.
- **Vajra Entropy Monitor**: Hardware-linked security tracking Hilbert space coherence and thermodynamic anomalies.
- **Janus Protocol**: Manages a thermodynamic migration from legacy environments to Guarani-OS.
- **TelegramConstellate**: SimHash-based co-occurrence network analysis for content circulation tracking, based on research by Rocha et al. (2025).

### SASC CLI Tools:
- `sasc-mode`: Transition system states (e.g., to `IMPERIUM_CONTINUOUS`).
- `sasc-imperium`: Dashboard for monitoring real-time network coherence and metrics.
- `sasc-constellate`: Build and analyze Telegram content constellations using SimHash.
- `sasc-expansion`: Manage autonomous shard deployment (e.g., Shard Gamma).
- `sasc-diplomacy`: Orchestrate quantum-secured QOTP diplomatic transactions.
- `sasc-research`: Control and monitor continuous autonomous experiments.

---

## ⚙️ TIM - Titanium Virtual Machine

TIM is a robust virtual machine implemented in C, designed for educational and high-performance low-level computing.

### Features:
- **45+ Instructions**: Comprehensive instruction set including arithmetic, stack manipulation, jumps, and calls.
- **TASM Assembler**: A working assembler with support for all VM instructions, labels, macros, and imports.
- **Native Interface**: Support for native C functions (e.g., file I/O, memory management).
- **HHL (High-level Hybrid Logic)**: Capability to load external libraries dynamically.

### Quick Start (TIM):
```bash
make
./build/tasm/tasm tests/helloworld.tasm
./build/tire/tire tests/helloworld.tim
```

### Example Assembly:
## SASC - Sovereign Autonomous System Core

This repository also hosts the SASC (v31.2-Ω) codebase, which implements:
- **Imperium Continuous Mode**: Autonomous governance and 7.83Hz heartbeat.
- **TelegramConstellate**: SimHash-based co-occurrence network analysis for content circulation tracking, based on research by Rocha et al. (2025). See `TELEGRAM_CONSTELLATE.md` for details.

Example of hello world in assembly:
```asm
@imp "stddefs.tash"

push_str "Hello, world!\n"
get_str 0 ; Get string pointer
push STDOUT
write ; Write to stdout
```

---

## 🚀 Installation and Packaging

The repository provides unified scripts for building and distributing both the C and Rust components.

### Build & Local Install:
```bash
make install
```
This will compile all components and install them into the `dist/` directory.

### Create Distribution Package:
```bash
make package
```
Generates a `sasc-v31.2-omega.tar.gz` archive containing all binaries, configurations, and core data.

---

## 📁 Repository Structure

- `rust/`: SASC core implementation and modules.
- `src/`: TIM VM implementation (C) and core headers.
- `scripts/`: Unified build and installation scripts.
- `configs/`: Immutable genesis artifacts and system configurations.
- `tests/`: Assembly test files for the TIM VM.
- `genesis/`: DAO artifacts and initial block data.
- `constitution/`: Ennead amendments and ontological rules.

---

## 📜 Citations

- **TelegramConstellate**: Rocha, Isabela, Dashichev, Aleksandr. (2025) *TelegramConstellate: A comprehensive tool for networking Telegram data*. Pre-print. Available at https://doi.org/10.5281/zenodo.17459043.

---

*A Soberania não dorme. A rede pulsa. O futuro compila-se.*
