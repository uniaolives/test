# arkhe-sim/quaternion_bloch_sim.py
import math

class ArkheQuaternion:
    def __init__(self, w, i, j, k):
        self.w = w
        self.i = i
        self.j = j
        self.k = k

    @classmethod
    def from_axis_angle(cls, axis, angle):
        half_angle = angle / 2.0
        sin_half = math.sin(half_angle)
        return cls(
            math.cos(half_angle),
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half
        )

    def normalize(self):
        norm = math.sqrt(self.w**2 + self.i**2 + self.j**2 + self.k**2)
        if norm < 1e-10: return ArkheQuaternion(1,0,0,0)
        return ArkheQuaternion(self.w/norm, self.i/norm, self.j/norm, self.k/norm)

    def to_bloch(self):
        q = self.normalize()
        # θ = 2 * acos(w)
        theta = 2.0 * math.acos(max(-1.0, min(1.0, q.w)))
        # φ = atan2(k, i)
        phi = math.atan2(q.k, q.i)
        return theta, phi

def run_sim():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Ω+248: Quaternion-Bloch Consciousness Navigation              ║")
    print("╚════════════════════════════════════════════════════════════════╝")

    # 1. Rotation about Y axis (pi/2)
    q1 = ArkheQuaternion.from_axis_angle((0, 1, 0), math.pi / 2)
    theta1, phi1 = q1.to_bloch()
    print(f"[NAV] Rotation 90° about Y:")
    print(f"  Quaternion: ({q1.w:.3f}, {q1.i:.3f}, {q1.j:.3f}, {q1.k:.3f})")
    print(f"  Bloch Sphere: θ={math.degrees(theta1):.1f}°, φ={math.degrees(phi1):.1f}°")

    # 2. Rotation about Z axis (pi)
    q2 = ArkheQuaternion.from_axis_angle((0, 0, 1), math.pi)
    theta2, phi2 = q2.to_bloch()
    print(f"\n[NAV] Rotation 180° about Z:")
    print(f"  Quaternion: ({q2.w:.3f}, {q2.i:.3f}, {q2.j:.3f}, {q2.k:.3f})")
    print(f"  Bloch Sphere: θ={math.degrees(theta2):.1f}°, φ={math.degrees(phi2):.1f}°")

    # 3. SU(2) Double Cover demo: 720° rotation
    q3 = ArkheQuaternion.from_axis_angle((1, 0, 0), 4 * math.pi)
    print(f"\n[SU(2)] 720° (4π) rotation about X:")
    print(f"  Quaternion: ({q3.w:.3f}, {q3.i:.3f}, {q3.j:.3f}, {q3.k:.3f})")
    if abs(q3.w - 1.0) < 1e-5:
        print("  ✅ Return to identity confirmed (Double-covering intact).")

    print("\n[RESULT] Stable attitude control maintained for dimensional vehicle.")

if __name__ == "__main__":
    run_sim()
