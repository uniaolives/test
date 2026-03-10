# 🜏 Debian Pi Arkhe: Deployment Guide (Standard Debian)

This guide provides instructions for deploying the OrbVM runtime and Teknet node components on a Raspberry Pi 5 (8GB) running standard Debian 12 (Bookworm) ARM64.

## Prerequisites

- **Hardware**: Raspberry Pi 5 (8GB recommended)
- **OS**: Debian 12 (Bookworm) ARM64
- **Software**: Rust 1.75+ (installed via rustup)
- **Dependencies**:
  ```bash
  sudo apt update
  sudo apt install build-essential libssl-dev pkg-config
  ```

## 1. Build OrbVM Runtime

Clone the repository and build the `orbvm-runtime` in release mode.

```bash
git clone https://github.com/arkhe-protocol/arkhe-os
cd arkhe-os
cargo build --release --bin orbvm-runtime
sudo cp target/release/orbvm-runtime /usr/bin/
```

## 2. Configure Systemd Service

Deploy the `orbvm.service` file to the systemd directory and enable it.

```bash
# Copy the service file
sudo cp systemd/orbvm.service /etc/systemd/system/

# Create the configuration directory
sudo mkdir -p /etc/orbvm
sudo touch /etc/orbvm/engine.conf

# Create the 'arkhe' user if it doesn't exist
sudo useradd -r -s /bin/false arkhe

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable orbvm
sudo systemctl start orbvm
```

## 3. Verify Installation

Check the status of the OrbVM service:

```bash
sudo systemctl status orbvm
```

## 4. Using orbctl

You can monitor and control the node using the `orbctl` CLI tool.

```bash
# Build orbctl
cd arkhe-linux/core/orbctl
cargo build --release
sudo cp target/release/orbctl /usr/local/bin/

# Query coherence
orbctl coherence

# Monitor coherence (watch mode)
orbctl coherence --watch

# Execute an Orb
orbctl execute --lambda 0.85 --phi 4.5
```

## Future Work: Kernel Integration

Current deployment operates in userspace. Future versions (Post-Week 12) will include kernel-level patches for `phase_t` and custom syscalls (`orb_evolve`, `orb_collapse`) to enable zero-copy phase field operations on the Raspberry Pi 5.

---
🜏 **Arkhe Protocol | Debian Pi Arkhe Roadmap**
