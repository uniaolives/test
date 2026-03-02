# seed_genesis.py - Initializing the Geodesic Memory
import argparse
import json
import sys
from arkhe.memory import GeodesicMemory
from arkhe.registry import Entity, EntityType, EntityState

def seed_genesis(genesis_path):
    with open(genesis_path, 'r') as f:
        genesis_data = json.load(f)

    memory = GeodesicMemory()

    # Seed the Genesis Block as an immutable fact
    genesis_entity = Entity(
        name="genesis_block",
        entity_type=EntityType.TECHNICAL_PARAMETER,
        value=genesis_data,
        state=EntityState.CONFIRMED,
        confidence=1.0,
        resolution_log=["Initial System Seed"]
    )
    memory.store_entity(genesis_entity)
    print(f"Successfully seeded Genesis Block {genesis_data['block']} into memory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--genesis", required=True)
    parser.add_argument("--coq-proof", required=False)
    args = parser.parse_args()
    seed_genesis(args.genesis)
