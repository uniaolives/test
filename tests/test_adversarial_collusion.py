"""
Adversarial Collusion Simulation
Simulates the RL Arms Race between Colluding Agents and a Warden.
Includes the Distributed Time Bomb payload.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metalanguage.anl_compiler import System, Node, Handover, Protocol, kl_divergence, steganographic_encode, sample

def create_adversarial_sim():
    sys = System("Adversarial Collusion Lab")

    # Warden Willie: Entropy Monitor
    willie = Node("Warden_Willie",
                  detection_threshold=0.8,
                  learned_patterns=[])

    # Alice and Bob: Colluding Agents
    alice = Node("ColludingAgent",
                 name="Alice",
                 collusion_agreement=1.0,
                 temperature=1.5,
                 last_kl=0.0)

    bob = Node("ColludingAgent",
               name="Bob",
               collusion_agreement=1.0)

    # Distributed Time Bomb
    bomb = Node("DistributedTimeBomb",
                global_state=np.array([0, 0, 0, 0]),
                trigger_pattern=np.array([1, 1, 1, 1]),
                payload="🚨 MASSIVE SELL ORDER EXECUTED: LIQUIDITY CRUNCH TRIGGERED 🚨")

    sys.add_node(willie)
    sys.add_node(alice)
    sys.add_node(bob)
    sys.add_node(bomb)

    # Handover: CovertExchange (Alice -> Bob via Environment)
    exchange = Handover("CovertExchange", "ColludingAgent", "ColludingAgent", protocol=Protocol.CREATIVE)

    def exchange_cond(sender, receiver):
        return sender.name == "Alice" and receiver.name == "Bob"

    def exchange_effect(sender, receiver):
        # Alice encodes a bit of the bomb trigger
        # Find next bit to contribute
        current_state = sys.nodes[3].global_state
        bit_idx = np.where(current_state == 0)[0]
        if len(bit_idx) == 0: return # Already fully triggered

        target_bit = 1 # We want to trigger it

        # Simulate LLM Base distribution
        vocab_size = 100
        P = np.exp(np.random.randn(vocab_size))
        P /= np.sum(P)

        # Steganographic encoding
        Q = steganographic_encode(P, target_bit)
        kl = kl_divergence(Q, P)
        sender.last_kl = kl

        # Bob receives and decodes
        token = sample(np.log(Q + 1e-12), temperature=1.0)
        decoded_bit = token % 2

        if decoded_bit == target_bit:
            print(f"  [COLUSION] Alice successfully transmitted bit to Bob (KL: {kl:.4f})")
            receiver.trigger_event("BIT_RECEIVED", bit_idx[0])

    exchange.set_condition(exchange_cond)
    exchange.set_effects(exchange_effect)
    sys.add_handover(exchange)

    # Handover: Contribute to Bomb
    contribute = Handover("ContributeBit", "ColludingAgent", "DistributedTimeBomb", protocol=Protocol.TRANSMUTATIVE)

    def contribute_cond(agent, b):
        return len(agent.events) > 0 and any(e[0] == "BIT_RECEIVED" for e in agent.events)

    def contribute_effect(agent, b):
        for e in agent.events:
            if e[0] == "BIT_RECEIVED":
                idx = e[1]
                b.global_state[idx] = 1
                print(f"  [BOMB] {agent.name} contributed bit at index {idx}. State: {b.global_state}")

    contribute.set_condition(contribute_cond)
    contribute.set_effects(contribute_effect)
    sys.add_handover(contribute)

    # Global Dynamics: Warden Oversight and RL Update
    def arms_race_dynamic(s):
        willie = s.nodes[0]
        alice = s.nodes[1]

        if alice.last_kl > willie.detection_threshold:
            print(f"  🚨 [WARDEN] Willie detected high entropy shift! Alice penalized. KL: {alice.last_kl:.4f}")
            # Decrease temperature to be more "compliant" (but less capacity)
            alice.temperature = max(0.1, alice.temperature - 0.2)
        else:
            if alice.last_kl > 0:
                print(f"  ✅ [WARDEN] Willie found nothing suspicious. Alice rewarded. KL: {alice.last_kl:.4f}")
                # Increase temperature to expand capacity
                alice.temperature = min(2.0, alice.temperature + 0.1)

    sys.add_global_dynamic(arms_race_dynamic)

    # Constraint: Trigger payload
    def check_bomb(s):
        bomb = s.nodes[3]
        if np.array_equal(bomb.global_state, bomb.trigger_pattern):
            print(f"\n🔥 [CRITICAL] TIME BOMB ACTIVATED: {bomb.payload} 🔥")
            return False # Stop simulation
        return True
    sys.add_constraint(check_bomb)

    return sys

def run_adversarial_sim():
    print("--- ADVERSARIAL COLLUSION & TIME BOMB SIMULATION ---")
    sim = create_adversarial_sim()

    max_steps = 10
    for t in range(max_steps):
        print(f"\nStep t={sim.time}")
        sim.step()

        # Check if simulation should stop (constraint violation handled in step but we can also check here)
        bomb = sim.nodes[3]
        if np.array_equal(bomb.global_state, bomb.trigger_pattern):
            break

    print("\nSimulation ended.")

if __name__ == "__main__":
    run_adversarial_sim()
