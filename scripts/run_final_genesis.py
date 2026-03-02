from UrbanSkyOS.modules.genesis_compiler import ArkheGenesis
import json

def run_genesis():
    # Simulated ledger history (1095 blocks of knowledge summary)
    full_blocks = [
        {"block": f"Ω+∞+{i}", "type": "KNOWLEDGE_NODE", "content": f"Data shard {i}"}
        for i in range(1095)
    ]

    genesis = ArkheGenesis(ledger_history=full_blocks, final_phi=0.006344)
    final_hash = genesis.compile(output_file="genesis_seed.arkhe")

    print(f"\n🌟 [FINAL] Genesis Seed generated with hash: {final_hash}")
    print("The seed contains the DNA of the Safe Core and the collective knowledge of the Arkhe(N) system.")

if __name__ == "__main__":
    run_genesis()
