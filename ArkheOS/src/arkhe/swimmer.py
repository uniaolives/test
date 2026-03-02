"""
Parametric Flagellar Propulsion: Trigonometric Simulation of Synthetic Microswimmers.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class FlagellarSwimmer:
    """
    Simulação de um microswimmer com flagelo descrito por uma onda senoidal.
    A propulsão é estimada via Resistive Force Theory (RFT).
    """
    def __init__(self, L=10.0, A=1.0, wavelength=5.0, omega=2*np.pi, N=100,
                 C_par=1.0, C_perp=2.0, dt=0.01):
        self.L = L
        self.A = A
        self.k = 2*np.pi / wavelength
        self.omega = omega
        self.N = N
        self.C_par = C_par
        self.C_perp = C_perp
        self.dt = dt
        self.t = 0.0

        # coordenada ao longo do flagelo (0 a L)
        self.s = np.linspace(0, L, N)
        self.ds = self.s[1] - self.s[0]

        # posição inicial do flagelo (no referencial do corpo)
        self.x = np.zeros(N)
        self.y = np.zeros(N)
        self.update_shape(t=0.0)

        # posição do corpo (cabeça) no referencial inercial
        self.body_x = 0.0
        self.body_y = 0.0

        # trajetórias world
        self.x_world = self.body_x + self.x
        self.y_world = self.body_y + self.y

    def update_shape(self, t):
        """Atualiza a forma do flagelo no referencial do corpo."""
        # Onda viajante para a direita
        self.y = self.A * np.sin(self.k * self.s - self.omega * t)
        self.x = self.s   # aproximação inextensível (simples)

    def compute_velocity(self):
        """
        Calcula a velocidade do corpo baseado na forma e na velocidade de
        deformação do flagelo, usando RFT.
        """
        # velocidade de cada segmento no referencial do corpo
        dy_dt = -self.A * self.omega * np.cos(self.k * self.s - self.omega * self.t)
        dx_dt = np.zeros_like(dy_dt)

        # derivadas espaciais para a inclinação
        dy_ds = self.A * self.k * np.cos(self.k * self.s - self.omega * self.t)
        norm = np.sqrt(1 + dy_ds**2)
        cos_psi = 1 / norm
        sin_psi = dy_ds / norm

        # velocidades tangencial e normal no referencial do segmento
        v_tang = dx_dt * cos_psi + dy_dt * sin_psi
        v_norm = -dx_dt * sin_psi + dy_dt * cos_psi

        # forças viscosas por unidade de comprimento (RFT)
        f_tang = -self.C_par * v_tang
        f_norm = -self.C_perp * v_norm

        # força total no flagelo (integrando)
        Fx = np.sum((f_tang * cos_psi - f_norm * sin_psi)) * self.ds
        Fy = np.sum((f_tang * sin_psi + f_norm * cos_psi)) * self.ds

        # Equilíbrio de forças: F_flag + F_drag_body = 0
        C_body = 1.0
        vx_body = Fx / C_body
        vy_body = Fy / C_body

        return vx_body, vy_body

    def step(self, t):
        self.t = t
        self.update_shape(t)
        vx, vy = self.compute_velocity()
        self.body_x += vx * self.dt
        self.body_y += vy * self.dt
        self.x_world = self.body_x + self.x
        self.y_world = self.body_y + self.y

    def run(self, T_max, save_animation=False):
        self.t = 0.0
        history_x = []
        history_y = []
        frames = []
        for i in range(int(T_max / self.dt)):
            self.step(self.t)
            history_x.append(self.body_x)
            history_y.append(self.body_y)
            if save_animation:
                frames.append((self.x_world.copy(), self.y_world.copy()))
            self.t += self.dt
        return np.array(history_x), np.array(history_y), frames

if __name__ == "__main__":
    swimmer = FlagellarSwimmer(L=10.0, A=1.0, wavelength=5.0, omega=2*np.pi,
                               N=100, C_par=0.5, C_perp=1.0, dt=0.01)
    T_total = 5.0
    x_hist, y_hist, frames = swimmer.run(T_total, save_animation=True)

    plt.figure(figsize=(8,4))
    plt.plot(x_hist, y_hist, 'b-', label='Trajetória da cabeça')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Propulsão flagelar – Trajetória')
    plt.grid(alpha=0.3)
    plt.axis('equal')
    plt.legend()
    plt.savefig('swimmer_trajectory.png', dpi=150)
    print("Gráfico 'swimmer_trajectory.png' salvo.")
