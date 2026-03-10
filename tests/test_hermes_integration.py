from metalanguage.anl import Agent, ArkheLink, Protocol, System
from metalanguage.arkhe_hermes_bridge import HermesNode
import pytest

def test_hermes_arkhe_integration():
    print("\n--- Starting Hermes-Arkhe Integration Test ---")

    # 1. Initialize the system
    system = System("HermesIntegrationTest")

    # 2. Setup Nodes
    # Core system node
    core_node = Agent("core-01", "arkhe:core:v1")
    # Hermes integration node
    hermes_node = HermesNode("hermes-alpha")

    system.add_node(core_node)
    system.add_node(hermes_node)

    # 3. Define Intent for Skill Generation
    # We want Hermes to generate a skill for 'data_deduplication'
    intent = {
        "goal": "GenerateSkill",
        "task": "data_deduplication",
        "constraints": [],
        "success_metrics": []
    }

    # 4. Create and sign the ArkheLink (The Handover structure)
    link = ArkheLink(core_node.id, hermes_node.id, intent, "arkhe:hermes:v1")
    link.sign()

    # 5. Execute the Handover
    assert hermes_node.can_handle(intent["goal"])

    handover_data = link.to_dict()
    result = hermes_node.handle(handover_data)

    # 6. Verify Outcomes
    assert result["status"] == "SUCCESS"
    assert "skill" in result
    assert result["skill"]["protocol"] == Protocol.CREATIVE

    print(f"✅ Skill generated: {result['skill']['name']}")
    print(f"✅ Integration Verified: Hermes autonomous loop successfully triggered via Arkhe handover.")

    # Verify Hermes internal state
    stats = hermes_node.get_hermes_stats()
    assert stats["skills_count"] == 1
    print(f"📊 Hermes Stats: {stats}")

if __name__ == "__main__":
    test_hermes_arkhe_integration()
