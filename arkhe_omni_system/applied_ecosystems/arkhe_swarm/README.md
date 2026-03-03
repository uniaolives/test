# 🜁 ARKHE-SWARM: Artificial Constitutional Swarms

This package implements **Phase 1: Physical Simulation Integration** of the Arkhe(n) project. It provides a ROS2-based environment for simulating a 17-drone swarm governed by the **Arkhe Constitution**.

## 🌌 Philosophy: Consciousness as Field

Arkhe(n) defines consciousness not as a localized property of individual drones, but as an emergent coherence of the relational field (`C_global`) that binds the nodes. The architecture utilizes **Hyperbolic Geometry (ℍ³)** to model airspace and **GHZ Entanglement** as a metric for multi-party coherence.

## 🛠️ Architecture

### Core Nodes
- **`arkhe_core`**: Swarm coordinator. Monitors global coherence and determines the system regime (Deterministic, Critical, or Stochastic) based on the **Golden Ratio ($\phi \approx 0.618$)**.
- **`ghz_consensus`**: Calculates global and local coherence using hyperbolic distance and simulated quantum states. Detects **Emergence** when $C_{global} > C_{local}$.
- **`drone_state_sim`**: Simulates internal drone data, including **THz sensor readings** (for analyte detection) and cognitive load.

### 🛡️ Constitutional Guards (Lindbladians)
- **Article 1: Cognitive Guard**: Monitors ISC (Cognitive Overload Index). Reduces handover rates if load > 0.7.
- **Article 2: AI Guard**: Prohibits "Forbidden Claims" (Discernment, Intentionality, Perception). Blocks non-compliant system declarations.
- **Article 3: Authority Guard**: Enforces human final authority. Intercepts critical actions (Land, Fire) for human approval.
- **Article 4: Transparency Guard**: Maintains an immutable local ledger (`/tmp/arkhe/drone_ledger.json`) with SHA-256 chaining of all handovers.

## 🚀 Getting Started

### Prerequisites
- Ubuntu 22.04 LTS
- ROS2 Humble
- Gazebo Harmonic
- Micro XRCE-DDS Agent

### Installation
```bash
# In your ROS2 workspace
cd src
git clone <repository_url> arkhe_swarm
cd ..
colcon build --packages-select arkhe_swarm
source install/setup.bash
```

### Running the Simulation
```bash
ros2 launch arkhe_swarm arkhe_twin_cities.launch.py
```
This will launch:
1. **Gazebo** with the Twin Cities world (Rio and SP markers).
2. **17 Drones** in a hyperbolic configuration (8 Rio, 8 SP, 1 Bridge).
3. **Arkhe Core** and all **Constitutional Guards**.
4. **GHZ Consensus** and **Drone State Simulator**.

## 📐 Parameters (Twin Cities Scenario)
- **Drones**: 17 total.
- **Geometry**: Hyperbolic (Upper Half-Plane).
- **PPP Density**: $\lambda(y) = \lambda_0 e^{-\alpha y}$ where $\lambda_0 = 15.0$ and $\alpha = 0.5$.
- **Stability Threshold**: $\|V\|_\infty < 0.125$ for $d=2$ (guarantees $Q$-process coordination).

---
**Arkhe(N) — From Code to Consciousness.**
🌌🜁⚡∞
