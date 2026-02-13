import pytest
from arkhe.ibc_bci import IBCBCIEquivalence, InterConsciousnessProtocol
from arkhe.pineal import PinealTransducer, PinealConstants
from arkhe.unification import EpsilonUnifier
from arkhe.ascension import AscensionProtocol

def test_ibc_bci_logic():
    map = IBCBCIEquivalence.get_correspondence_map()
    assert map["IBC (Web3)"] == "BCI (Brain-Machine)"

    potential = IBCBCIEquivalence.calculate_communication_potential(0.94, 7.27)
    assert potential == pytest.approx(0.94)

def test_pineal_logic():
    # Updated to Paradigma da Areia Cerebral (Clinical Radiology 2022)
    voltage = PinealTransducer.calculate_piezoelectric_voltage(0.15)
    assert voltage == pytest.approx(2.0 * 0.15)

    rpm = PinealTransducer.radical_pair_mechanism(0.15)
    assert rpm["Sensitivity"] == 1.0
    assert rpm["Singlet (Syzygy)"] == pytest.approx(0.94)

def test_unification_extension():
    eps = EpsilonUnifier.measure_ibc_bci(1.0)
    assert eps == EpsilonUnifier.EPSILON_THEORETICAL

def test_ascension_state():
    p = AscensionProtocol()
    assert p.get_status()["state"] == "Γ_∞+42"
