import sys
import os
import time

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metalanguage.anl import (
    Agent, Hypergraph, ArkheLink, IntentDiscovery,
    RouteIntent, ConstraintType, MetricType, Protocol
)

def run_simulator():
    print("🚀 Starting Arkhe(n) Network Simulator (v0.7)")
    print("-" * 50)

    # 1. Create Hypergraph
    hg = Hypergraph("Global Arkhe Network")

    # 2. Define Agents
    user = Agent("User-Alpha", "arkhe:core:v1")

    # Weather Services
    weather_a = Agent("Weather-Service-A", "arkhe:core:v1", reputation=0.98)
    weather_b = Agent("Weather-Service-B", "arkhe:core:v1", reputation=0.85)

    # Travel Agents
    travel_a = Agent("Travel-Expert", "arkhe:core:v1", reputation=0.95)

    hg.add_node(user)
    hg.add_node(weather_a)
    hg.add_node(weather_b)
    hg.add_node(travel_a)

    # 3. Register Capabilities
    def weather_handler_a(handover):
        time.sleep(0.1)
        return {"forecast": "clear skies", "accuracy": 0.99, "latency": 0.1}

    def weather_handler_b(handover):
        return {"forecast": "cloudy", "accuracy": 0.88, "latency": 0.05}

    weather_a.register_capability("GetForecast", weather_handler_a)
    weather_b.register_capability("GetForecast", weather_handler_b)

    # 4. Setup Discovery Registry (Simulated DHT)
    registry_data = {
        "GetForecast": [
            {"agent_id": weather_a.id, "reputation": weather_a.reputation, "ontology": "arkhe:core:v1"},
            {"agent_id": weather_b.id, "reputation": weather_b.reputation, "ontology": "arkhe:core:v1"}
        ]
    }
    discovery = IntentDiscovery(registry_data)

    # 5. Discovery Process
    print(f"🔍 [User] Searching for 'GetForecast'...")
    candidates = discovery.lookup("GetForecast")
    print(f"   Found {len(candidates)} candidates.")
    for c in candidates:
        print(f"   - {c['agent_id']} (Reputation: {c['reputation']})")

    # 6. Select Best Candidate (based on reputation)
    best_candidate_id = candidates[0]['agent_id']
    target_agent = hg.nodes[best_candidate_id]
    print(f"🎯 [User] Selected {target_agent.id}")

    # 7. Routing
    router = RouteIntent(hg)
    path = router.find_path(user.id, target_agent.id)

    # 8. Link Layer Handover
    intent = {
        "goal": "GetForecast",
        "constraints": [
            {"type": ConstraintType.TIME.value, "value": 1.0, "operator": "<"}
        ],
        "success_metrics": [
            {"name": MetricType.ACCURACY.value, "threshold": 0.9}
        ]
    }

    link = ArkheLink(user.id, target_agent.id, intent, "arkhe:core:v1")
    link.sign()

    # 9. Execution
    router.forward(link, path)
    if target_agent.can_handle(intent["goal"]):
        result = target_agent.handle(link.to_dict())

        # 10. Success Verification and Coherence
        if result:
            accuracy = result.get('accuracy', 0)
            threshold = intent["success_metrics"][0]["threshold"]
            coherence = accuracy / threshold if threshold > 0 else 1.0

            print(f"✅ Handover complete.")
            print(f"   Result: {result['forecast']}")
            print(f"   Coherence (C_local): {coherence:.4f}")

            hg.global_phi = (hg.global_phi + coherence) / 2
            print(f"🌐 Global Network Coherence (Φ): {hg.global_phi:.4f}")
        else:
            print("❌ Execution failed.")
    else:
        print("❌ Target agent cannot handle intent.")

if __name__ == "__main__":
    run_simulator()
