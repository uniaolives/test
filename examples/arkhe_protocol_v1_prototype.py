import sys
import os
import numpy as np

# Add parent directory to path to import metalanguage
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metalanguage.anl import System, Node, ArkheLink, IntentObject, Constraint, Metric, ContextSnapshot

def run_protocol_simulation():
    print("🚀 ArkheProtocol v1 (Link Layer) Prototype Simulation\n")

    sys_web4 = System("Web4_Infrastructure_Simulator")

    # 1. Define Alice (Source)
    alice = Node("HumanAgent",
                 display_name="Alice",
                 capabilities=["INTENT_INITIATION"])

    # 2. Define WeatherService (Target)
    weather_service = Node("ServiceAgent",
                           display_name="SkySense Weather",
                           capabilities=["WEATHER_FORECAST", "DATA_PROVISION"])

    sys_web4.add_node(alice)
    sys_web4.add_node(weather_service)

    # 3. Discovery Phase (Network Layer)
    print("Step 1: Intent Discovery")
    goal = "WEATHER_FORECAST"
    potential_targets = sys_web4.discover_agents(goal)

    if not potential_targets:
        print("No agents found for goal:", goal)
        return

    target = potential_targets[0]
    print(f"  Found agent: {target.display_name} ({target.id}) for goal: {goal}\n")

    # 4. Handover Creation (Link Layer)
    print("Step 2: Link Layer Handover Initiation")

    # Define Intent
    accuracy_metric = Metric("ForecastAccuracy", threshold=0.9)
    time_constraint = Constraint("latency", value=500, operator="<") # ms

    intent = IntentObject(
        goal=goal,
        constraints=[time_constraint],
        success_metrics=[accuracy_metric]
    )

    # Define Context
    context = ContextSnapshot(
        source_state=hashlib.sha256(b"alice_state_t0").hexdigest(),
        ambient_conditions={"trust_score": 0.98}
    )

    # Instantiate ArkheLink
    link = ArkheLink(
        source=alice,
        target=target,
        intent=intent,
        context=context,
        ontology="arkhe:core:v1"
    )

    # 5. Define Preconditions, Effects, and Postconditions
    def alice_has_credits(source, target):
        return True # Mock credit check

    link.preconditions.append(alice_has_credits)

    def weather_forecast_effect(source, target):
        print(f"  [Effect] {target.display_name} generating forecast for {source.display_name}...")
        # Simulate data transfer
        source.received_data = {"forecast": "Sunny", "temp": 25}
        # Update metrics
        accuracy_metric.value = 0.95 # Simulated outcome

    link.set_effects(weather_forecast_effect)

    def source_received_data(source, target):
        return hasattr(source, 'received_data')

    link.postconditions.append(source_received_data)

    # 6. Execute Handover
    success = link.execute()

    # 7. System Evolution (Ouroboros)
    print("\nStep 3: System Evolution")
    sys_web4.step()
    print(f"  System Coherence: {sys_web4.coherence:.4f}")
    print(f"  Alice's Curiosity: {alice.curiosity_score:.4f}")

    if success:
        print("\n✅ Handover successful! Web4 Link Layer validated.")
    else:
        print("\n❌ Handover failed.")

import hashlib
if __name__ == "__main__":
    run_protocol_simulation()
