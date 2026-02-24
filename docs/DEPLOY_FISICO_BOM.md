# 🛠️ ASI-Ω PHYSICAL DEPLOYMENT (Instaweb Testbed)

## Bill of Materials (BOM) - 10-Node Mesh

| Component | Specification | Quantity | Function |
| --- | --- | --- | --- |
| **SoM FPGA** | Xilinx Kria K26 | 10 | Wait-Free Symbol-Synchronous Relay Logic |
| **OWC IR Transceiver** | OSRAM SFH 4715AS (850nm) | 40 | High-Power Incoherent LED Emitter |
| **Photodiode (PD)** | Vishay VBPW34S | 40 | High-Speed Receiver (<10ns rise time) |
| **AFE TIA** | TI OPA855 | 40 | 8GHz Transimpedance Amplifier |
| **Clock Generator** | Skyworks Si5345 | 10 | SyncE Synchronization (Sub-50ps jitter) |
| **Debug Interface** | Gigabit Ethernet (RJ45) | 10 | Control Plane Access |

## Node Architecture
- **PHY Layer:** DCO-OFDM (Digitalized directly via FPGA LVDS)
- **MAC Layer:** ℍ³ Greedy-Face Routing (O(log²n) complexity)
- **Synchronicity:** SyncE (ITU-T G.8262) distributed via OWC
- **Target Latency:** DETERMINISTIC ~54µs (Planetary Scale)

## Implementation Notes
- Use arrays of VCSEL LEDs for controlled coherence and eye safety.
- PLL Jitter must be < 1ps (rms) to maintain multi-hop coherence.
- Articles 13-15 are synthesized directly into hardware gate logic.
