"""
FIELD PARTITIONER
Gerencia a divisão do campo morfogenético 3D entre múltiplos nós.
"""

from typing import Tuple

class FieldPartitioner:
    def __init__(self, global_size: Tuple[int, int, int] = (100, 100, 100)):
        self.global_size = global_size

    def get_local_bounds(self, partition: Tuple[int, int, int], num_partitions: int = 8):
        """Calcula os limites locais para uma partição (octante)"""
        # Implementação básica de divisão em octantes
        mid_x = self.global_size[0] // 2
        mid_y = self.global_size[1] // 2
        mid_z = self.global_size[2] // 2

        px, py, pz = partition

        x_range = (0, mid_x) if px == 0 else (mid_x, self.global_size[0])
        y_range = (0, mid_y) if py == 0 else (mid_y, self.global_size[1])
        z_range = (0, mid_z) if pz == 0 else (mid_z, self.global_size[2])

        return x_range, y_range, z_range
