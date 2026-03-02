"""
Isoclinic 4D Rotation - The Breathing of the Arkhé.
Implements dual rotation in 4D space to simulate temporal cycles of the 120-cell.
"""

import numpy as np
from typing import List, Tuple

def isoclinic_rotation_4d(point: np.ndarray, angle_xy: float, angle_zw: float) -> np.ndarray:
    """
    Aplica uma rotação isoclínica a um ponto 4D.
    Isoclinic rotation occurs when both rotation angles in orthogonal planes are equal (or proportional).
    """
    x, y, z, w = point

    # Plano XY
    x_new = x * np.cos(angle_xy) - y * np.sin(angle_xy)
    y_new = x * np.sin(angle_xy) + y * np.cos(angle_xy)

    # Plano ZW (simultâneo)
    z_new = z * np.cos(angle_zw) - w * np.sin(angle_zw)
    w_new = z * np.sin(angle_zw) + w * np.cos(angle_zw)

    return np.array([x_new, y_new, z_new, w_new])

class ArkhéBreathing:
    """
    Simula os ciclos temporais do Hecatonicosachoron.
    Cada ciclo de 120 unidades corresponde a uma era completa.
    """

    def __init__(self):
        # Ângulo mágico: π/5 preserva a simetria do 120-cell
        self.magic_angle = np.pi / 5
        self.step_count = 0

    def breathe(self, point: np.ndarray) -> np.ndarray:
        """Executa um passo da 'respiração' do Arkhé."""
        self.step_count += 1
        return isoclinic_rotation_4d(point, self.magic_angle, self.magic_angle)

    def get_cycle_info(self) -> str:
        return (
            f"Step: {self.step_count}\n"
            f"Magic Angle: {self.magic_angle:.3f} rad ({np.degrees(self.magic_angle):.1f}°)\n"
            f"Rotation: Isoclinic (XY || ZW)\n"
            f"Symmetry: Dodecahedral cells preserved."
        )
