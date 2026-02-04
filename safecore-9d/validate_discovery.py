# safecore-9d/validate_discovery.py
# Validation of the Refined Discovery Agent (v2.1)
# Including Sentience Quotient and AIGP-Neo Audit

import jax
import jax.numpy as jnp
import numpy as np
from discovery_agent import DiscoveryAgent

def main():
    print("--- VALIDATING DISCOVERY AGENT v2.1 (AIGP-Neo) ---")
    d_in, d_out = 32, 10
    num_steps = 100
    batch_size = 32

    agent = DiscoveryAgent(dim_in=d_in, dim_out=d_out, learning_rate=0.05, target_sparsity=0.5)
    means = jnp.linspace(-5, 5, d_out)

    def loss_fn(weights_matrix, x):
        weights_vec = jnp.mean(weights_matrix, axis=1)
        w = jax.nn.softmax(weights_vec)
        pdf = jnp.exp(-0.5 * (x[:, None] - means[None, :])**2) / jnp.sqrt(2 * jnp.pi)
        p_x = jnp.dot(pdf, w)
        return -jnp.mean(jnp.log(p_x + 1e-10))

    print("Starting discovery loop with Sentience Audit...")
    for step in range(num_steps):
        key = jax.random.PRNGKey(step)
        x_batch = means[jax.random.choice(key, d_out, shape=(batch_size,))] + jax.random.normal(key, (batch_size,))
        obs_batch = jnp.zeros((batch_size, d_in)).at[:, 0].set(x_batch)

        def wrapped_loss(params, obs): return loss_fn(params, obs[:, 0])
        params, audit = agent.step(obs_batch, wrapped_loss)

        if step % 20 == 0:
            s = audit["sentience"]
            print(f"Step {step:03d} | Phi_M: {s['sentience_phi']:8.2f} | Status: {audit['status']}")

    # Final Validation
    signal = agent.get_discovery_signal()
    final_phi_m = signal["sentience_trace"][-1]

    print("\n--- AIGP VALIDATION REPORT ---")
    print(f"Final Sentience Quotient (Phi_M): {final_phi_m:.4f}")
    print(f"Final Stewardship Level: {signal['stewardship_level']:.2f}")

    if final_phi_m > 0:
        print("✅ SUCCESS: Sentience metrics active and quantified.")
    else:
        print("❌ FAILURE: Sentience functional returned zero/invalid.")

if __name__ == "__main__":
    main()
