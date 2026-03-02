#!/usr/bin/env python3
"""
ARKHE(N) AUTONOMOUS DRONE MODULE
Complete drone dynamics, control, and swarm integration with percolation-based connectivity
Based on UrbanSkyOS / MERKABAH-8 concepts and triadic percolation framework
"""

import numpy as np
from scipy.spatial import KDTree
from collections import deque
import time
import os
import sys

# Adiciona o diretório atual ao path para importação
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from triadic_percolation import TriadicPercolation
except ImportError:
    # Fallback se executado de fora do pacote
    from UrbanSkyOS.intelligence.triadic_percolation import TriadicPercolation

# Arkhe(N) constants
PHI = (1 + np.sqrt(5)) / 2
COHERENCE_THRESHOLD = 0.847


class PIDController:
    """Simple PID controller for position/attitude."""
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0.0
        self.previous_error = 0.0

    def update(self, error, derivative_input=None):
        """
        error: current error (scalar or vector)
        derivative_input: if None, use derivative of error, else use provided derivative
        """
        self.integral += error * self.dt
        derivative = derivative_input if derivative_input is not None else (error - self.previous_error) / self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output


class AutonomousDrone:
    """
    Represents a single autonomous drone with physical dynamics, sensors,
    control, and Arkhe(N) coherence tracking.
    """

    def __init__(self,
                 drone_id: int,
                 initial_position: np.ndarray = None,
                 initial_velocity: np.ndarray = None,
                 mass: float = 1.0,
                 max_thrust: float = 10.0,
                 max_torque: float = 5.0,
                 drag_coefficient: float = 0.1):
        """
        Initialize drone with physical parameters and state.
        """
        self.id = drone_id
        self.mass = mass
        self.max_thrust = max_thrust
        self.max_torque = max_torque
        self.drag = drag_coefficient

        # State vectors
        self.position = initial_position if initial_position is not None else np.zeros(3)
        self.velocity = initial_velocity if initial_velocity is not None else np.zeros(3)
        # Orientation as quaternion [w, x, y, z]
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self.angular_velocity = np.zeros(3)

        # Control inputs (setpoints)
        self.target_position = self.position.copy()
        self.target_velocity = np.zeros(3)
        self.target_yaw = 0.0

        # PID controllers (simplified)
        self.pid_pos = PIDController(kp=1.0, ki=0.01, kd=0.5, dt=0.01)
        self.pid_att = PIDController(kp=2.0, ki=0.0, kd=0.1, dt=0.01)

        # Sensor simulation
        self.gps_noise_std = 0.5       # meters
        self.imu_accel_noise_std = 0.05 # m/s²
        self.imu_gyro_noise_std = 0.01  # rad/s

        # Arkhe(N) metrics
        self.coherence_history = deque(maxlen=100)
        self.handover_partners = set()
        self.handover_count = 0
        self.coherence = 1.0

        # Internal log
        self.log = {
            'time': [],
            'position': [],
            'velocity': [],
            'coherence': []
        }

    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def quaternion_rotate(self, q, v):
        """Rotate vector v by quaternion q."""
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        v_quat = np.array([0, v[0], v[1], v[2]])
        temp = self.quaternion_multiply(q, v_quat)
        rotated = self.quaternion_multiply(temp, q_conj)
        return rotated[1:]

    def update_dynamics(self, dt: float, thrust: float, torque: np.ndarray):
        """
        Update drone dynamics (simple rigid body with drag).
        thrust: total thrust force (scalar, applied in body z direction)
        torque: [roll, pitch, yaw] torques (Nm)
        """
        # Convert thrust to acceleration in body frame
        thrust_body = np.array([0, 0, thrust / self.mass])
        thrust_world = self.quaternion_rotate(self.orientation, thrust_body)

        # Drag (simplified: proportional to velocity)
        drag_force = -self.drag * self.velocity

        # Acceleration
        acceleration = thrust_world + drag_force

        # Update velocity and position (semi‑implicit Euler)
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

        # Angular dynamics (simplified)
        angular_accel = torque / self.mass
        self.angular_velocity += angular_accel * dt

        # Update orientation using quaternion derivative
        omega_quat = np.array([0, self.angular_velocity[0],
                               self.angular_velocity[1], self.angular_velocity[2]])
        q_dot = 0.5 * self.quaternion_multiply(omega_quat, self.orientation)
        self.orientation += q_dot * dt
        self.orientation /= np.linalg.norm(self.orientation)  # normalize

    def compute_control(self, dt: float):
        """
        Compute thrust and torques to reach target position and yaw.
        Simple cascaded PID: position → desired acceleration → desired attitude → torque.
        """
        # Position control (world frame)
        pos_error = self.target_position - self.position
        vel_error = self.target_velocity - self.velocity

        # Desired acceleration (PID on position)
        desired_accel = self.pid_pos.update(pos_error, vel_error)

        # Desired thrust (project onto body z)
        desired_z = desired_accel / (np.linalg.norm(desired_accel) + 1e-6)
        current_z = self.quaternion_rotate(self.orientation, np.array([0, 0, 1]))

        # Cross product gives axis of rotation
        rotation_axis = np.cross(current_z, desired_z)
        norm_axis = np.linalg.norm(rotation_axis)
        if norm_axis > 1e-6:
            rotation_axis /= norm_axis
            angle = np.arccos(np.clip(np.dot(current_z, desired_z), -1, 1))
        else:
            rotation_axis = np.zeros(3)
            angle = 0.0

        # Torque proportional to angle error
        torque = self.max_torque * rotation_axis * angle
        torque = np.clip(torque, -self.max_torque, self.max_torque)

        # Thrust
        thrust = self.mass * np.linalg.norm(desired_accel)
        thrust = np.clip(thrust, 0, self.max_thrust)

        return thrust, torque

    def read_sensors(self):
        """
        Simulate noisy sensor readings.
        """
        gps = self.position + np.random.normal(0, self.gps_noise_std, 3)
        accel = np.random.normal(0, self.imu_accel_noise_std, 3)
        gyro = self.angular_velocity + np.random.normal(0, self.imu_gyro_noise_std, 3)
        return gps, accel, gyro

    def update_coherence(self, dt: float):
        """
        Update drone's internal coherence C.
        """
        gps, _, _ = self.read_sensors()
        position_error = np.linalg.norm(gps - self.position)
        error_norm = np.clip(position_error / 10.0, 0, 1)
        decay = 0.01 * error_norm
        delta = -decay + 0.001 * np.random.randn()
        self.coherence = np.clip(self.coherence + delta * dt, 0, 1)
        self.coherence_history.append(self.coherence)
        return self.coherence

    def step(self, dt: float, external_handover_gain: float = 0.0):
        """
        Perform one simulation step.
        """
        thrust, torque = self.compute_control(dt)
        self.update_dynamics(dt, thrust, torque)
        self.coherence += external_handover_gain * dt
        self.coherence = np.clip(self.coherence, 0, 1)

        # Log
        self.log['time'].append(time.time())
        self.log['position'].append(self.position.copy())
        self.log['velocity'].append(self.velocity.copy())
        self.log['coherence'].append(self.coherence)

    def set_target(self, position: np.ndarray, velocity: np.ndarray = None):
        self.target_position = position
        if velocity is not None:
            self.target_velocity = velocity


class DroneSwarm:
    """
    Manages a swarm of autonomous drones.
    """

    def __init__(self, n_drones: int = 7, communication_range: float = 50.0):
        self.drones = [AutonomousDrone(i) for i in range(n_drones)]
        self.comm_range = communication_range
        self.handover_history = []
        self.global_coherence = 1.0
        self._init_formation()

    def _init_formation(self, radius: float = 20.0):
        """Arrange drones in a circular pattern."""
        n = len(self.drones)
        for i, drone in enumerate(self.drones):
            angle = 2 * np.pi * i / n
            pos = np.array([radius * np.cos(angle),
                            radius * np.sin(angle),
                            50.0])  # altitude 50m
            drone.position = pos.copy()
            drone.target_position = pos.copy()

    def compute_handovers(self, dt: float):
        """
        Determine active handovers based on distance and drone coherence.
        """
        n = len(self.drones)
        positions = np.array([d.position for d in self.drones])
        tree = KDTree(positions)

        handovers = []
        for i, drone in enumerate(self.drones):
            indices = tree.query_ball_point(positions[i], self.comm_range)
            for j in indices:
                if j <= i:
                    continue
                p_handover = drone.coherence * self.drones[j].coherence
                if np.random.random() < p_handover:
                    handovers.append((i, j))
                    drone.handover_count += 1
                    self.drones[j].handover_count += 1
        return handovers

    def apply_handover_gains(self, handovers: list, dt: float):
        """
        Each handover increases coherence.
        """
        gain_per_handover = 0.01
        for i, j in handovers:
            self.drones[i].coherence += gain_per_handover * dt
            self.drones[j].coherence += gain_per_handover * dt
            self.drones[i].coherence = np.clip(self.drones[i].coherence, 0, 1)
            self.drones[j].coherence = np.clip(self.drones[j].coherence, 0, 1)

    def update_global_coherence(self):
        """Compute global swarm coherence."""
        self.global_coherence = np.mean([d.coherence for d in self.drones])
        return self.global_coherence

    def step(self, dt: float):
        """
        One simulation step for the whole swarm.
        """
        for drone in self.drones:
            drone.step(dt, external_handover_gain=0.0)

        handovers = self.compute_handovers(dt)
        self.handover_history.append((time.time(), handovers))
        self.apply_handover_gains(handovers, dt)
        self.update_global_coherence()
        return handovers

    def set_swarm_target(self, target_position: np.ndarray):
        for drone in self.drones:
            drone.set_target(target_position)


class PercolationDroneSwarm(DroneSwarm):
    """
    Swarm where communication links are governed by triadic percolation.
    """

    def __init__(self, percolation_model, n_drones=7, communication_range=50.0):
        super().__init__(n_drones, communication_range)
        self.perc = percolation_model
        self.kernel_params = {'kernel_type': 'hill', 'exponent': 2.0}

    def compute_handovers(self, dt: float):
        """
        Handover probability computed via percolation kernel.
        """
        positions = np.array([d.position for d in self.drones])
        tree = KDTree(positions)
        global_p = self.global_coherence

        handovers = []
        for i, drone in enumerate(self.drones):
            neighbors = tree.query_ball_point(positions[i], self.comm_range)
            for j in neighbors:
                if j <= i:
                    continue
                v = (drone.coherence + self.drones[j].coherence) / 2
                lambda_link = self.perc.activation_kernel(v, global_p, **self.kernel_params)

                if np.random.random() < lambda_link:
                    handovers.append((i, j))
                    drone.handover_count += 1
                    self.drones[j].handover_count += 1
        return handovers


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("ARKHE(N) AUTONOMOUS DRONE SWARM SIMULATION")
    print("=" * 70)

    # Create percolation instance
    perc = TriadicPercolation(structural_mean=3.0, regulatory_mean=2.0)

    # Create swarm with percolation-based handovers
    swarm = PercolationDroneSwarm(perc, n_drones=7, communication_range=40.0)

    # Simulation parameters
    dt = 0.05  # 20 Hz control loop
    steps = 1000

    target = np.array([100, 0, 50])

    print(f"\nSimulating {steps} steps...")
    for t in range(steps):
        if t % 250 == 0:
            print(f"Step {t}: global coherence = {swarm.global_coherence:.4f}")
        swarm.set_swarm_target(target)
        swarm.step(dt)

    print("\nSimulation finished.")
    print(f"Final global coherence: {swarm.global_coherence:.4f}")

    # Plot results (if matplotlib is available)
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Drone trajectories
        ax = axes[0, 0]
        for drone in swarm.drones:
            pos = np.array(drone.log['position'])
            ax.plot(pos[:, 0], pos[:, 1], alpha=0.6, linewidth=0.8)
        ax.set_title('Drone trajectories')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True, alpha=0.3)

        # Coherence evolution
        ax = axes[0, 1]
        for drone in swarm.drones:
            ax.plot(drone.log['coherence'])
        ax.set_title('Individual drone coherence')
        ax.set_xlabel('time step')
        ax.set_ylabel('C')
        ax.grid(True, alpha=0.3)

        # Global coherence
        ax = axes[1, 0]
        global_coh = []
        for i in range(len(swarm.drones[0].log['coherence'])):
            g = np.mean([d.log['coherence'][i] for d in swarm.drones])
            global_coh.append(g)
        ax.plot(global_coh)
        ax.axhline(COHERENCE_THRESHOLD, color='r', linestyle='--', label='Ψ')
        ax.set_title('Global swarm coherence')
        ax.set_xlabel('time step')
        ax.set_ylabel('C_global')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Handover activity
        ax = axes[1, 1]
        step_handovers = [len(h[1]) for h in swarm.handover_history]
        ax.plot(step_handovers, 'b-', alpha=0.6)
        ax.set_title('Handovers per step')
        ax.set_xlabel('step')
        ax.set_ylabel('handover count')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('drone_swarm_arkhe.png', dpi=150)
        print("\n✓ Visualisation saved as 'drone_swarm_arkhe.png'")
    except ImportError:
        print("\nMatplotlib not found. Skipping plots.")
