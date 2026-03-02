import sys
import os
import time

# Add metalanguage to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metalanguage.arkhe_human_tool import Human, Tool, InteractionGuard

def test_human_tool_logic():
    print("Testing Human-Tool Interface Logic...")

    # 1. Setup
    human = Human(processing_capacity=500, attention_span=30)
    tool = Tool(output_volume=200, output_entropy=2.5) # load = (200 * 2.5) / 500 = 1.0

    guard = InteractionGuard(human, tool)

    # 2. Test Blocked (Overload)
    print("  Checking if overload is blocked...")
    output = guard.propose_interaction("Write a book")
    assert output is None
    assert guard.log[-1]['event'] == 'BLOCKED'
    assert guard.log[-1]['reason'] == 'cognitive_overload'
    print("  ✅ Overload blocked successfully")

    # 3. Test Allowed (Safe Load)
    print("  Checking if safe load is allowed...")
    tool.output_volume = 100 # load = (100 * 2.5) / 500 = 0.5
    output = guard.propose_interaction("Write a poem")
    assert output is not None
    assert "poem" in output
    assert guard.log[-1]['event'] == 'GENERATED'
    current_load_after = human.current_load
    assert current_load_after > 0
    print(f"  ✅ Safe load allowed. Current human load: {current_load_after:.2f}")

    # 4. Test Review
    print("  Checking review mechanism...")
    guard.review(output, approved=True)
    assert guard.log[-1]['event'] == 'REVIEWED'
    assert guard.log[-1]['approved'] == True
    assert human.current_load < current_load_after
    print(f"  ✅ Review processed. Current human load: {human.current_load:.2f}")

    # 5. Test Cumulative Overload
    print("  Checking cumulative human overload...")
    human.current_load = 0.9
    output = guard.propose_interaction("Another task")
    assert output is None
    assert guard.log[-1]['reason'] == 'human_overloaded'
    print("  ✅ Cumulative overload blocked successfully")

    # 6. Test Metrics
    print("  Checking metrics...")
    idx = guard.cognitive_load_index()
    tpa = guard.authorship_loss_rate()
    print(f"  ISC (Cognitive Load Index): {idx:.2f}")
    print(f"  TPA (Authorship Loss Rate): {tpa:.2f}")
    assert idx > 0
    assert tpa > 0
    print("  ✅ Metrics calculated successfully")

if __name__ == "__main__":
    try:
        test_human_tool_logic()
        print("\n🏆 ALL TESTS PASSED!")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
