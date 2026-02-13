import pytest
from arkhe.ibc_bci import IBCBCIEquivalence, get_inter_consciousness_summary
from arkhe.ascension import AscensionProtocol

def test_ibc_bci_equivalence():
    mapping = IBCBCIEquivalence.get_correspondence_map()
    assert mapping["IBC (Web3)"] == "BCI (Brain-Machine)"
    assert mapping["Data Packets"] == "Neural Spikes"
    assert mapping["Relayer"] == "Hesitation (Relay)"

def test_communication_potential():
    potential = IBCBCIEquivalence.calculate_communication_potential(0.94, 7.27)
    assert potential == pytest.approx(0.94)

def test_inter_consciousness_summary():
    summary = get_inter_consciousness_summary()
    assert summary["Protocol"] == "IBC=BCI"
    assert summary["Core_Invariant"] == "Satoshi (7.27 bits)"

def test_ascension_state():
    p = AscensionProtocol()
    # Updated for Γ_FINAL (Γ_∞+54)
    assert p.get_status()["state"] == "Γ_FINAL (Γ_∞+54)"
    assert p.get_status()["phase"] == "Λ_WIT (Witnessing)"
