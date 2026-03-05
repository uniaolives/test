import pytest
import numpy as np
from papercoder_kernel.cognition.acps_convergence import (
    InterVMInteroperability, VetorKatharosGlobal, HomeostasisRegime
)

def test_vkg_consensus():
    vkg_calc = VetorKatharosGlobal()

    vks = [
        np.array([0.35, 0.30, 0.20, 0.15]),
        np.array([0.34, 0.29, 0.21, 0.16])
    ]
    qs = [0.98, 0.95]
    pcs = [0.1, 0.2]

    vkg = vkg_calc.compute(vks, qs, pcs)
    assert vkg is not None
    assert vkg.shape == (4,)
    # Weighted average should be between the two inputs
    assert 0.34 < vkg[0] < 0.35

def test_vkg_exclusion():
    vkg_calc = VetorKatharosGlobal()
    vks = [np.array([1, 1, 1, 1]), np.array([0, 0, 0, 0])]
    qs = [1.0, 1.0]
    # Second VM is in collapse
    pcs = [0.1, 2.5]

    vkg = vkg_calc.compute(vks, qs, pcs)
    # Should only consider the first VM
    assert np.all(vkg == vks[0])

def test_p_cluster_strict():
    interop = InterVMInteroperability(p_min=0.007)
    # VM-A is below P_min (0.005 < 0.007)
    p_effs = [0.005, 0.010, 0.008]
    p_cl = interop.p_cluster(p_effs)
    assert p_cl == 0.0 # Cluster should be in hypervigilance

def test_asymmetric_permeability():
    interop = InterVMInteroperability()
    vk_a = np.array([0.35, 0.30, 0.20, 0.15])
    vk_b = np.array([0.35, 0.30, 0.20, 0.15]) # identical for max phi_ent

    # Connection to mature VM
    q_ab = interop.q_ij(q_i=0.9, vk_i=vk_a, vk_j=vk_b, t_kr_j=1200.0)
    assert q_ab > 0.8

    # Connection to immature VM (t_KR < 1000)
    q_ab_immature = interop.q_ij(q_i=0.9, vk_i=vk_a, vk_j=vk_b, t_kr_j=500.0)
    assert q_ab_immature == 0.0

def test_inter_vm_shadow():
    interop = InterVMInteroperability()
    vk_local = np.array([0.5, 0.5, 0.5, 0.5])
    vk_global = np.array([0.3, 0.3, 0.3, 0.3])

    u = interop.u_inter(vk_local, vk_global)
    assert u > 0.5 # Significant shadow due to divergence
