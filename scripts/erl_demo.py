# scripts/erl_demo.py
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.abspath("src"))

from papercoder_kernel.merkabah.self_node import SelfNode
from papercoder_kernel.cognition.erl import ExperientialLearning
from papercoder_kernel.cognition.utils import Memory, Environment

logging.basicConfig(level=logging.INFO)

def main():
    print("--- Experiential Reinforcement Learning (ERL) Demo ---")

    # Setup
    self_node = SelfNode()
    memory = Memory()
    env = Environment()
    erl = ExperientialLearning(self_node, memory, env, threshold=0.5)

    # Run episode
    input_task = "decipher_HT88"
    print(f"\nProcessing Task: {input_task}")
    result = erl.episode(input_task)

    print("\nEpisode Results:")
    print(f"  Attempt 1: {result['y1']} (Reward: {result['r1']})")
    if result['y2']:
        print(f"  Reflection: {result['delta']}")
        print(f"  Attempt 2: {result['y2']} (Reward: {result['r2']})")
        print(f"  Reflection Reward: {result['reward_reflect']}")
    else:
        print("  Attempt 1 was successful, no reflection needed.")

    print(f"\nMemory Storage: {memory.retrieve()}")

if __name__ == "__main__":
    main()
