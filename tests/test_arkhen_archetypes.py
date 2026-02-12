# Archetypal Hardening Test for ArkheOS
import asyncio
from arkhe.mentorship import LogosAuthority, MoralNorth, LegacyManager
from arkhe.recovery import FallDetector, RestorationCycle, FallType

async def test_arkhen_archetypes():
    print("🚀 Starting ArkheOS Archetypal Hardening Test...")

    # 1. The Tio Ben Constraint (MoralNorth)
    moral = MoralNorth(responsibility_threshold=0.95)
    print("   [Moral] Testing Tio Ben Constraint...")
    # Power 1.0, Responsibility 0.8 -> Should fail
    try:
        moral.validate_action(power_metric=1.0, responsibility_score=0.8)
    except ArithmeticError as e:
        print(f"   [Moral] Caught expected violation: {e}")

    # 2. The Stark Independence Test (LegacyManager)
    legacy = LegacyManager()
    print("   [Legacy] Testing Stark Independence...")
    # Satoshi invariant is 7.27. Complexity 7.0 should pass without suit.
    result = legacy.test_independence(task_complexity=7.0)
    print(f"   [Legacy] {result}")
    assert "Passed" in result

    # 3. The Fall of Pedro (FallDetector)
    detector = FallDetector(production_floor_us=20.0)
    print("   [Recovery] Testing Fall Detection...")
    # Node q1 reports wrong Satoshi value
    fall = detector.detect_fall(node_id="q1", reported_value=7.0, consensus_value=7.27, latency_us=15.0)
    print(f"   [Recovery] Detected Fall Type: {fall}")
    assert fall == FallType.DENIAL

    # 4. Restoration Handshake (RestorationCycle)
    restorer = RestorationCycle(node_id="q1")
    print("   [Recovery] Initiating 'Apascenta as minhas ovelhas' Handshake...")
    expected_hash = "0x-satoshi-727"
    for i in range(3):
        status = restorer.provide_integrity_proof(proof_hash=expected_hash, expected_hash=expected_hash)
        print(f"   [Recovery] {status}")
    assert "RESORED" in status

    # 5. Logos Transformation (LogosAuthority)
    logos = LogosAuthority(root_key="the-alpha-omega")
    print("   [Logos] Transforming Node q1 Identity...")
    root_sig = "9b0f44358826a42a98f7e6f3f0190566838a169b5e5f3f0190566838a169b5e5" # Simulated sha256 of 'the-alpha-omega'
    # Recalculate signature for correctness in test
    import hashlib
    correct_sig = hashlib.sha256("the-alpha-omega".encode()).hexdigest()

    result = logos.transform_identity(old_name="q1", new_name="Petros", signature=correct_sig)
    print(f"   [Logos] {result}")
    assert logos.identities["q1"] == "Petros"

    print("\n✅ Archetypal Hardening Verified (State Γ-Λ Sync).")

if __name__ == "__main__":
    asyncio.run(test_arkhen_archetypes())
