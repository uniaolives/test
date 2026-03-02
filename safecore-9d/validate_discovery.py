# safecore-9d/validate_discovery.py
# Validation of the Refined Discovery Agent (v2.1)
# Including Sentience Quotient and AIGP-Neo Audit
# Validation of the Refined Discovery Agent (v2.0)
# Toy Problem: Online Learning of Gaussian Mixture Weights (d=10)

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
import matplotlib.pyplot as plt

def main():
    print("--- VALIDATING DISCOVERY AGENT v2.0 ---")
    d_in = 32 # Input dimension (for K-FAC activations simulation)
    d_out = 10 # Output dimension (GMM parameters)
    num_steps = 200
    batch_size = 32

    # Initialize Discovery Agent
    agent = DiscoveryAgent(dim_in=d_in, dim_out=d_out, learning_rate=0.05, target_sparsity=0.5)

    # Ground Truth: 10-component GMM weights
    true_weights = jax.random.dirichlet(jax.random.PRNGKey(42), jnp.ones(d_out))
    means = jnp.linspace(-5, 5, d_out)

    def loss_fn(weights_matrix, x):
        # x is a batch of scalar observations (batch_size,)
        # weights_matrix is [d_out, d_in]. We collapse it to a d_out vector.
        weights_vec = jnp.mean(weights_matrix, axis=1)
        w = jax.nn.softmax(weights_vec)
        # Probability density: p(x) = sum(w_i * N(x|mu_i, 1))
        # pdf shape: (batch_size, d_out)
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
    # Simulation loop
    print(f"Starting discovery loop for {num_steps} steps...")
    for step in range(num_steps):
        # Generate streaming data (the 'x' to be fitted)
        key = jax.random.PRNGKey(step)
        component = jax.random.choice(key, d_out, shape=(batch_size,), p=true_weights)
        noise = jax.random.normal(key, (batch_size,))
        x_batch = means[component] + noise

        # Discovery Step
        # The agent.step uses observations for both loss_fn and K-FAC.
        # So we pass x_batch but the agent expects dim_in=32.
        # We'll just pad x_batch or use it to generate obs_batch.
        # For simplicity, we'll redefine obs_batch to be (batch_size, d_in)
        # where the first column is x_batch.
        obs_batch = jnp.zeros((batch_size, d_in))
        obs_batch = obs_batch.at[:, 0].set(x_batch)

        # Update loss_fn to only use the first column of x
        def wrapped_loss(params, obs):
            return loss_fn(params, obs[:, 0])

        params, audit = agent.step(obs_batch, wrapped_loss)

        if step % 20 == 0:
            print(f"Step {step:03d} | Status: {audit['status']} | Loss: {agent.history['loss'][-1]:.4f}")

    # Results Validation
    learned_weights = jax.nn.softmax(jnp.mean(agent.params, axis=1))
    weight_error = jnp.linalg.norm(learned_weights - true_weights)

    print("\n--- VALIDATION REPORT ---")
    print(f"Weight Reconstruction Error: {weight_error:.6f}")

    if weight_error < 0.6:
        print("✅ SUCCESS: Discovery Agent v2.0 converged.")
    else:
        print("❌ FAILURE: Discovery failed to converge.")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(agent.history['loss'])
    plt.title("Discovery Agent v2.0: Convergence")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("discovery_regret_v2.png")
    print("Plot saved to discovery_regret_v2.png")

if __name__ == "__main__":
    main()
