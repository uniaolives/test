# tests/test_noesis_integration.py
import asyncio
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.noesis.ceo import NOESISCEOAgent
from modules.noesis.oversoul import CorporateOversoul
from modules.noesis.orchestration import NOESISAgentOrchestration
from modules.noesis.types import CorporateDecision

async def test_ceo_agent():
    print("--- Testing NOESIS CEO-Agent ---")
    ceo = NOESISCEOAgent(jurisdiction="Zug, Switzerland", initial_capital=1_000_000_000)

    # 1. Test Breathing
    print("\nTesting breathing cycle...")
    for _ in range(3):
        ceo.breathe_step()

    # 2. Test Strategic Decision
    print("\nTesting strategic decision...")
    situation = {"market_trend": "bullish", "competitor_action": "expansion"}
    decision = await ceo.strategic_decision(situation)

    if decision:
        print(f"Decision made: {decision.id} - {decision.content}")
        assert decision.id in [0, 1, 2] or isinstance(decision.id, str)
    else:
        print("Decision failed (as expected if gates blocked it)")

async def test_oversoul():
    print("\n--- Testing Corporate Oversoul ---")
    oversoul = CorporateOversoul(initial_capital=500_000_000, jurisdiction="Singapore")
    oversoul.breathe()

    decision = oversoul.strategic_decision({"goal": "expand_to_mars"})
    print(f"Oversoul decision: {decision.content if decision else 'None'}")

async def test_orchestration():
    print("\n--- Testing Agent Orchestration ---")
    orch = NOESISAgentOrchestration()
    plan = orch.coordinate_agents("Q3_Financial_Optimization")
    print(f"Orchestrated plan: {plan.content}")
    assert "plan_CEO" in plan.id

async def main():
    try:
        await test_ceo_agent()
        await test_oversoul()
        await test_orchestration()
        print("\n✅ NOESIS CORP Integration Tests Passed!")
    except Exception as e:
        print(f"\n❌ Integration Tests Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
