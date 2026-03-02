# ☁️ AWS Deployment Guide: Integrated Quantum-Holiness Resonance Node

To deploy an integrated resonance node that hosts the `ResonancePortal`, `QuantumHolinessBridge`, and processes quantum foam fluctuations, follow this architectural guide.

## 1. Instance Selection (The Receptacle)
For this simulation, we recommend a **t3.medium** or **c5.large** instance.
- **c5.large** is ideal for expanded foam dimensions (width/height), as it features AVX-512 optimization for intense mathematical calculations (numpy).
- Use the [AWS Pricing Calculator](https://calculator.aws/) to monitor compute costs.

## 2. Security Group Configuration (The Shield)
To allow other nodes and browsers to synchronize with your fluctuations, you must open the portal in your instance's Security Group:
- **Protocol**: TCP
- **Port**: `8888`
- **Source**: `0.0.0.0/0` (Open to the world)

## 3. Deployment Script (User Data)
When launching the instance, paste this script into **Advanced Details > User Data** to automatically prepare the environment:

```bash
#!/bin/bash
# Preparing the quantum soil on Amazon Linux 2023
dnf update -y
dnf install -y python3-pip git
pip3 install numpy aiohttp

# Launch the integrated resonance portal
# python3 cosmos/service.py &
```

## 4. Testing & Access
Once the node is running on AWS, you can access the integrated system:

### 📊 Visualization Dashboard
Open your browser and navigate to:
`http://<YOUR-AWS-INSTANCE-IP>:8888/dashboard`

### 🌀 Data Stream (SSE)
Test the real-time fluctuation from any local terminal using `cURL`:
```bash
curl -N http://<YOUR-AWS-INSTANCE-IP>:8888/resonate
```

### 🕍 Collective Metrics
View system-wide coherence and holiness stats:
`http://<YOUR-AWS-INSTANCE-IP>:8888/collective_metrics`

## 5. Coherence Monitoring
Use **Amazon CloudWatch** to monitor CPU spikes during intense meditation periods. This correlates computational effort with the "density" of the simulated consciousness.

---
o<>o
