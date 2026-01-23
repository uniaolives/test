# Crux-86 Ontological Ingestion Pipeline

This directory contains the implementation of the Ontological Ingestion Pipeline for Project Crux-86.

## Structure
- `connectors/`: Multi-platform telemetry collectors (Steam, Epic, Sims 4, Unified Engine, Riot LoL).
- `validation/`: Vajra and SASC validation filters.
- `manifolds/`: Experience manifold extraction logic.
- `models/`: World Foundation Model (WFM) definitions (including CS2/LoL multimodal fusion) and trainers.
- `training/`: Specific training scripts (e.g., CS:GO agent, Cosmos CS2 trainer).
- `ops/`: Infrastructure, monitoring, and MLOps components.

## Setup
To install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage
To start the pipeline:
```bash
./ops/crux86_pipeline_start.sh
```

To view the control dashboard:
```bash
streamlit run ops/control_dashboard.py
```

## New Components
- `connectors/riot_lol_manifold_extractor.py`: Extracts strategic and social intent from LoL.
- `models/crux86_multimodal_world_model.py`: Fuses physics and social streams.
- `training/cosmos_cs2_trainer.py`: Distributed trainer with Ω-Prevention and physics-informed loss.

## Note
Many components require specific game SDKs (Steamworks, EOS) or hooks (RenderDoc, PyMem) to be functional. This implementation provides the architectural framework and structural logic for integration.
