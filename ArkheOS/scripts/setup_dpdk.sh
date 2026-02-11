# scripts/setup_dpdk.sh
#!/bin/bash
set -e

# 1. Instala DPDK 23.11 LTS
wget https://fast.dpdk.org/rel/dpdk-23.11.tar.xz
tar xf dpdk-23.11.tar.xz
cd dpdk-23.11

# 2. Compila
meson build
ninja -C build
ninja -C build install

# 3. Setup Hugepages (2GB)
echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

# 4. Bind NIC ao driver UIO
modprobe uio_pci_generic
dpdk-devbind.py --bind=uio_pci_generic 0000:01:00.0

# 5. Valida
dpdk-testpmd -l 0-1 -n 4 -- -i
