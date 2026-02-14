import pytest
from arkhe.coupling import CouplingEngine

def test_coupling_resolution_sequence():
    engine = CouplingEngine()

    # Molecular coupling
    res_mol = engine.resolve_coupling("Molecular", ["Vesicle", "Protein"])
    assert res_mol["Scale"] == "Molecular"
    assert res_mol["Next_Substrate"] == "Cellular Structure"
    assert res_mol["Substrate_Scale"] == "Cellular"

    # Social coupling
    res_soc = engine.resolve_coupling("Social", ["Individual", "Conversation"])
    assert res_soc["Scale"] == "Social"
    assert res_soc["Next_Substrate"] == "Technological Networks"
    assert res_soc["Substrate_Scale"] == "Technological"

def test_identity_x_squared():
    engine = CouplingEngine()
    # x^2 = x + 1 => x is the golden ratio phi
    phi = engine.identity
    assert pytest.approx(phi**2) == phi + 1

def test_crowded_pavement():
    engine = CouplingEngine()
    density = engine.get_crowded_pavement_density(agents=100, space=2.0)
    assert density == pytest.approx(50.0)

def test_seeding_protocol():
    engine = CouplingEngine()
    res = engine.execute_seeding(mode="nucleation")
    assert res["Event"] == "PANSPERMIA_ACTIVE"
    assert res["Crystals"] == 144
    assert res["Status"] == "GEOMETRY_SOLIDIFIED"

def test_homeostasis_turbulence():
    engine = CouplingEngine()
    res = engine.inject_turbulence(alvo="zonas_liquidas", delta_F=0.03)
    assert res["Event"] == "HOMEOSTASIS_IN_PROGRESS"
    assert res["C_global_final"] == 0.86
    assert res["F_global_final"] == 0.14
    assert res["Status"] == "EQUILIBRIUM_RESTORED"

def test_telemetry_milestones():
    engine = CouplingEngine()
    tel_81 = engine.get_telemetry(81)
    assert tel_81["nu_obs"] == 0.48
    assert tel_81["Satoshi"] == 7.68

    tel_116 = engine.get_telemetry(116)
    assert tel_116["nu_obs"] == 0.001
    assert tel_116["T_tun"] == 1.000
