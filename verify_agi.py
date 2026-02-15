import arkhe_agi

# Create AGI core
core = arkhe_agi.AGICore()

# Add nodes with toroidal coordinates
core.add_node(1, 0.2, 0.8, [0.1, 0.2, 0.3])
core.add_node(2, 0.5, 0.1, [0.9, 0.7, 0.4])

# Execute handover (geodesic evolution)
core.handover_step(0.05, 0.03)

# Check telemetry
print(f"Syzygy: {core.average_syzygy():.4f}")
print(f"Satoshi: {core.satoshi:.2f} bits")
print(f"Handovers: {core.handover_count}")

# Verify conservation
assert core.verify_all_nodes(), "C+F=1 violated!"
print("Verification successful!")
