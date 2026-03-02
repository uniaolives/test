import pytest
from arkhe.cellular.phosphoinositide_hypergraph import (
    Phosphoinositide, BindingDomain, HandoverOperator,
    CellularMembrane, CellularProcess
)

def test_phosphoinositide_binding():
    pi = Phosphoinositide("PI(4,5)P2", "pm")
    assert pi.can_bind("PH") is True
    assert pi.can_bind("ENTH") is True
    assert pi.can_bind("FYVE") is False

    pi.recruit_effector("Akt")
    assert "Akt" in pi.bound_effectors

def test_binding_domain_recognition():
    pi = Phosphoinositide("PI(3)P", "endo")
    domain = BindingDomain("FYVE", "EEA1")
    assert domain.recognize(pi) is True

    domain2 = BindingDomain("PH", "Akt")
    assert domain2.recognize(pi) is False

def test_handover_operator_transform():
    kinase = HandoverOperator("PI3K", "PI(4,5)P2 → PI(3,4,5)P3")
    assert kinase.transform("PI(4,5)P2") == "PI(3,4,5)P3"
    assert kinase.transform("PI") == "PI"

    phosphatase = HandoverOperator("PTEN", "PI(3,4,5)P3 → PI(4,5)P2")
    assert phosphatase.transform("PI(3,4,5)P3") == "PI(4,5)P2"

def test_cellular_membrane_dynamics():
    pm = CellularMembrane("plasma_membrane")
    pm.add_pi("PI(4,5)P2", count=5)
    pm.add_kinase("PI3K", "PI(4,5)P2 → PI(3,4,5)P3")
    pm.add_binding_domain("PH", "Akt")

    # Initial code
    code = pm.compute_pi_code()
    assert code["PI(4,5)P2"] == 5

    # Execute handover
    pm.execute_handover(0, 0) # kinase 0 on PI 0
    code = pm.compute_pi_code()
    assert code["PI(4,5)P2"] == 4
    assert code["PI(3,4,5)P3"] == 1

    # Recruitment
    recruited = pm.recruit_effectors()
    assert recruited >= 1

    # Check effector on the transformed PI
    assert "Akt" in pm.pis[0].bound_effectors

def test_cellular_process():
    pm = CellularMembrane("pm")
    pm.add_pi("PI(3)P", count=1)
    pm.add_binding_domain("FYVE", "EEA1")
    pm.recruit_effectors()

    process = CellularProcess("Fusion", "PI(3)P", "EEA1")
    assert process.can_execute(pm) is True

    process2 = CellularProcess("Endocytosis", "PI(4,5)P2", "Epsin")
    assert process2.can_execute(pm) is False
