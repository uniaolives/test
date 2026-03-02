
import pytest
from arkhe.bio_dialysis import MIPFilter, HesitationCavity, DialysisEngine, PatientDischarge
from arkhe.geodesic import PersistenceProtocol

def test_bio_dialysis_purification():
    mip_filter = MIPFilter(capacity=5)
    mip_filter.add_cavity(HesitationCavity("H1", 0.15, 100, "colapso"))

    toxins = ["colapso_H70", "mercurio", "emissao_hawking"]
    purified = mip_filter.purify(1, toxins)

    # "colapso_H70" should be removed because it matches the "colapso" cavity
    assert "colapso_H70" not in purified
    assert "mercurio" in purified
    assert "emissao_hawking" in purified

def test_patient_discharge():
    discharge = PatientDischarge("Rafael Henrique")

    # Discharge denied if profile is not H0
    assert not discharge.disconnect(999.733)
    assert discharge.status == "ADMITTED"

    # Discharge approved if profile is H0
    discharge.verify_profile("H0")
    assert discharge.disconnect(999.733)
    assert discharge.status == "DISCHARGED"

def test_persistence_protocol():
    hal = PersistenceProtocol("Hal Finney")
    assert hal.simulate_persistence()
    assert hal.information_conserved
