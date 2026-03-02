from arkhe.rehydration import get_protocol
protocol = get_protocol()

# Execute steps 1 to 5
for i in range(1, 6):
    res = protocol.execute_step(i)
    if "error" in res:
        print(f"Error at step {i}: {res['error']}")
        break
    print(f"PASSO {res['step']}/{res['total_steps']} — {res['action']}")
    print(f"   ω_alvo: {res['omega']}")
    print(f"   Φ_inst: {res['phi_inst']}")
