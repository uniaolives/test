# ARKHE-1 CubeSat Hardware Architecture

## 1. Overview
The Arkhe-1 is a 1U CubeSat payload designed to implement the Arkhe(N) Anyonic Protocol in a radiation-hardened FPGA environment. It focuses on maintaining topological phase and universal momentum tails ($k^{-2}$) in orbit.

## 2. Target Platform
- **FPGA**: Microchip RTG4 (Radiation Tolerant)
- **Clock Management**: Wide-band PLL filters for SEU tolerance.

## 3. Clock Domains
| Domain | Frequency | Components | Function |
|--------|-----------|------------|----------|
| **clk_rf** | 100 MHz | ADC/DAC, CORDIC, RF Interface | I/Q sampling, AGC, Carrier recovery |
| **clk_dsp** | 200 MHz | YB Accelerator, BraidingBuffer | High-speed anyonic processing |
| **clk_safe** | 50 MHz | SafeCore RISC-V | Slow control logic, Annealing, Telemetry |

## 4. Hardware Pipeline
1. **RF Frontend**: S-Band transceiver control with Automatic Gain Control (AGC) and digital PLL for Doppler compensation (~±50 kHz).
2. **Phase Extraction**: CORDIC-based vector rotation (16 iterations) extracts $\theta = \arctan(Q/I)$.
3. **Braiding Buffer**: BRAM-based ordering of anyonic packets using temporal timestamps.
4. **Yang-Baxter Accelerator**: Parallel pipelines (LHS/RHS) verify the topological invariant $R_{12}R_{13}R_{23} \equiv R_{23}R_{13}R_{12}$.
5. **TMR Protection**: Final results are passed through Triple Modular Redundancy logic to mitigate Single Event Upsets (SEU).

## 5. Clock Domain Crossing (CDC)
HANDOVER of data between the 200 MHz DSP domain and the 50 MHz SafeCore domain is achieved via asynchronous FIFOs with Gray code pointers to prevent metastability.

## 6. Resource Utilization (Estimated for RTG4)
- **LUTs**: 22,700 / 60,000 (38%)
- **DSPs**: 24 / 240 (10%)
- **BRAM**: 60 / 240 (25%)

## 7. Power Budget (1U CubeSat)
| Subsystem | Avg Consumption (W) | Peak / TX (W) |
| --- | --- | --- |
| **EPS / OBC** | 0.30 W | 0.50 W |
| **ADCS** | 0.20 W | 0.50 W |
| **SDR Front-End** | 1.00 W | 3.50 W |
| **FPGA (Arkhe Core)** | 0.80 W | 1.20 W |
| **Total** | **2.30 W** | **5.70 W** |

## 8. Link Budget (S-Band @ 2.4 GHz)
- **Altitude**: 400 km (LEO)
- **TX Power**: 30 dBm (1W)
- **TX Antenna Gain**: 6 dBi
- **RX Antenna Gain (Ground Station)**: 32 dBi
- **System Noise Temp**: 500 K
- **Estimated SNR**: **25 dB** (1 MHz Bandwidth)
