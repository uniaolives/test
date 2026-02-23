from arkhe.rehydration import get_protocol
protocol = get_protocol()

# Simulate progress to step 2
protocol.execute_step(1)
protocol.execute_step(2)

# Execute requested step 3
res = protocol.execute_step(3)
print(f"PASSO {res['step']}/{res['total_steps']} — {res['action']}")
print(f"ω_alvo: {res['omega']}")
print(f"Φ_inst: {res['phi_inst']}")
print(f"Status: {res['status']}")
