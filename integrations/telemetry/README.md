# Crux-86 Ontological Ingestion Pipeline

This directory contains the implementation of the Ontological Ingestion Pipeline for Project Crux-86.

## Structure
- `connectors/`: Multi-platform telemetry collectors (Steam, Epic, Sims 4, Unified Engine, Riot LoL, AoE II Governance).
- `validation/`: Vajra and SASC validation filters, substrate consistency checks.
- `manifolds/`: Experience manifold extraction logic.
- `models/`: World Foundation Model (WFM) definitions, MAE (Economic Attention Mechanism), optimizations, and trainers.
- `training/`: Specific training scripts (e.g., CS:GO agent, Cosmos CS2 trainer).
- `ops/`: Infrastructure, monitoring, MLOps components, and Civilizational Stability Monitors.

## Setup
To install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage
To start the pipeline (Simulation Mode):
```bash
./ops/crux86_pipeline_start.sh
```

To view the control dashboard:
```bash
streamlit run ops/control_dashboard.py
```

## Key Components
- `connectors/aoe_governance_connector.py`: Extracts governance manifolds from AoE II replays.
- `models/economic_attention_mechanism.py`: Sparse attention mechanism (MAE) for civilizational scale.
- `ops/SASCStabilityMonitor.py`: Real-time monitoring of the Benevolence Index (β).
- `validation/substrate_consistency_validator.py`: Ensures macro governance commands respect micro physical limits.

## Note
Many components require specific game SDKs (Steamworks, EOS) or hooks (RenderDoc, PyMem) to be functional. This implementation provides the architectural framework and structural logic for integration.
