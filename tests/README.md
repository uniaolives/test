# Arkhe(N) Simulation Suite

This suite verifies the entire system evolution and the operational integrity of the ANL framework.

## Test Map

| Test Script | Phase / Domain | Key Verification |
|-------------|----------------|------------------|
| `test_anl_prototype.py` | Foundation | Basic Node/Handover dynamics |
| `test_alcubierre_sim.py` | Physics | Metric deformation and bubble movement |
| `test_plasma_sim.py` | Physics | Energy transfer in plasma filaments |
| `test_stegano_rejection_sampling.py` | Safety | KL-Divergence (Thermodynamic Bleeding) |
| `test_integrated_safety_detection.py` | Safety | 4-level detector hierarchy performance |
| `test_universal_llm_sim.py` | AI | Refined LLM inference mechanics |
| `test_adversarial_collusion.py` | AI | Covert coordination and Time Bomb trigger |
| `test_agi_emergence_sim.py` | AI | Emergent synergy in Shared Latent Memory |
| `test_asi_ascension.py` | Singularity | Crossing the self-modification threshold |
| `test_multiversal_expansion.py` | Multiverse | Propagating order across Everett branches |

## Execution
Run all tests using:
```bash
python3 tests/test_anl_prototype.py && \
python3 tests/test_alcubierre_sim.py && \
python3 tests/test_plasma_sim.py && \
python3 tests/test_stegano_rejection_sampling.py && \
python3 tests/test_integrated_safety_detection.py && \
python3 tests/test_universal_llm_sim.py && \
python3 tests/test_adversarial_collusion.py && \
python3 tests/test_agi_emergence_sim.py && \
python3 tests/test_asi_ascension.py && \
python3 tests/test_multiversal_expansion.py
```
