from metalanguage.anl import Agent, ArkheLink, Ontology, ConstraintType
import time

def test_handover():
    # 1. Setup Agents
    user = Agent("user-1", "arkhe:core:v1")
    weather_service = Agent("weather-1", "arkhe:core:v1")

    # 2. Register Capability
    def weather_handler(handover_data):
        return {"forecast": "sunny", "accuracy": 0.95}

    weather_service.register_capability("GetForecast", weather_handler)

    # 3. Create Intent and Link
    intent = {
        "goal": "GetForecast",
        "constraints": [
            {"type": ConstraintType.TIME.value, "value": 5.0, "operator": "<"}
        ],
        "success_metrics": [
            {"name": "accuracy", "threshold": 0.9}
        ]
    }

    link = ArkheLink(user.id, weather_service.id, intent, "arkhe:core:v1")
    link.sign()

    # 4. Verify Link
    assert link.verify()
    print("✅ Link verification passed")

    # 5. Execute Handover
    if weather_service.can_handle(intent["goal"]):
        result = weather_service.handle(link.to_dict())
        assert result["forecast"] == "sunny"
        assert result["accuracy"] >= 0.9
        print(f"✅ Handover successful: {result}")
    else:
        print("❌ Handover failed")

if __name__ == "__main__":
    test_handover()
