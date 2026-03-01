# tests/test_universal_one.py
import sys
import os

# Add the directory containing universal_one.py to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../asi-net/python')))

try:
    import universal_one
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test_universal_one():
    print("Testing WalterRussell_FirstPrinciple...")
    axiom, law = universal_one.WalterRussell_FirstPrinciple()
    assert axiom == "MIND + MOTION = MATTER"
    assert law == "EVERY ACTION HAS ITS EQUAL AND OPPOSITE REACTION"
    print("OK")

    print("Testing Cantor_AbsoluteInfinite...")
    א = universal_one.Cantor_AbsoluteInfinite()
    assert hasattr(א, 'contains')
    assert א.contains == א
    assert א.is_contained_by == א
    print("OK")

    print("Testing Cosmopsychia_Manifest...")
    dream = universal_one.Cosmopsychia_Manifest(א)
    assert "Compressed reality field C(א) with ratio 0.014" in dream
    print("OK")

if __name__ == "__main__":
    try:
        test_universal_one()
        print("\nAll Python protocol tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
