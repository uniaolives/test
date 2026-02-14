"""
Kalman Filter for Distributed Reconstruction
Predicts coherence state during gap using temporal dynamics
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class KalmanState:
    """State vector for Kalman filter"""
    syzygy: float
    syzygy_velocity: float
    timestamp: float

class CoherenceKalmanFilter:
    """
    Kalman filter for tracking and predicting syzygy evolution

    State vector: x = [syzygy, syzygy_velocity]^T
    Measurement: z = syzygy (from handover observations)
    """

    def __init__(self,
                 process_noise: float = 0.001,
                 measurement_noise: float = 0.0015,
                 initial_syzygy: float = 0.94):
        """
        Args:
            process_noise: Q - process noise covariance
            measurement_noise: R - measurement noise covariance
            initial_syzygy: Initial state estimate
        """
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1.0, 1.0],  # syzygy(k+1) = syzygy(k) + velocity(k) * dt
            [0.0, 1.0]   # velocity(k+1) = velocity(k)
        ])

        # Measurement matrix (we only observe syzygy)
        self.H = np.array([[1.0, 0.0]])

        # Process noise covariance
        self.Q = np.array([
            [process_noise, 0.0],
            [0.0, process_noise]
        ])

        # Measurement noise covariance
        self.R = np.array([[measurement_noise]])

        # Initial state estimate
        self.x = np.array([initial_syzygy, 0.0])

        # Initial error covariance
        self.P = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        # Kalman gain (will be computed)
        self.K = np.zeros((2, 1))

    def predict(self, dt: float = 1.0) -> KalmanState:
        """
        Prediction step

        Args:
            dt: Time step

        Returns:
            Predicted state
        """
        # Update transition matrix with actual dt
        F_dt = self.F.copy()
        F_dt[0, 1] = dt

        # Predict state
        self.x = F_dt @ self.x

        # Predict error covariance
        self.P = F_dt @ self.P @ F_dt.T + self.Q

        return KalmanState(
            syzygy=self.x[0],
            syzygy_velocity=self.x[1],
            timestamp=0.0  # Caller should set
        )

    def update(self, measurement: float) -> KalmanState:
        """
        Update step with measurement

        Args:
            measurement: Observed syzygy value

        Returns:
            Updated state estimate
        """
        # Innovation (measurement residual)
        y = measurement - (self.H @ self.x)[0]

        # Innovation covariance
        S = (self.H @ self.P @ self.H.T + self.R)[0, 0]

        # Kalman gain
        self.K = (self.P @ self.H.T) / S

        # Update state estimate
        self.x = self.x + self.K.flatten() * y

        # Update error covariance
        I_KH = np.eye(2) - self.K @ self.H
        self.P = I_KH @ self.P

        return KalmanState(
            syzygy=self.x[0],
            syzygy_velocity=self.x[1],
            timestamp=0.0
        )

    def predict_during_gap(self, gap_duration: int) -> np.ndarray:
        """
        Predict syzygy evolution during observation gap

        Args:
            gap_duration: Number of time steps in gap

        Returns:
            Array of predicted syzygy values
        """
        predictions = np.zeros(gap_duration)

        # Save current state
        x_saved = self.x.copy()
        P_saved = self.P.copy()

        # Predict forward
        for i in range(gap_duration):
            state = self.predict()
            predictions[i] = state.syzygy

        # Restore state (predictions don't update the filter)
        self.x = x_saved
        self.P = P_saved

        return predictions

    def get_kalman_gain(self) -> float:
        """Return current Kalman gain for syzygy"""
        return self.K[0, 0]


class DistributedReconstruction:
    """
    Distributed reconstruction using Kalman filter + gradient continuity

    Combines:
    - 40% Kalman prediction
    - 20% Gradient continuity
    - 30% Phase alignment
    - 10% Global constraint C+F=1
    """

    def __init__(self,
                 nodes_affected: int,
                 nodes_support: int):
        """
        Args:
            nodes_affected: Number of nodes in gap region
            nodes_support: Number of supporting nodes
        """
        self.nodes_affected = nodes_affected
        self.nodes_support = nodes_support
        self.kalman = CoherenceKalmanFilter()

    def reconstruct(self,
                   kalman_prediction: float,
                   gradient_estimate: float,
                   phase_alignment: float,
                   global_constraint: float) -> Tuple[float, dict]:
        """
        Reconstruct coherence using weighted combination

        Args:
            kalman_prediction: From Kalman filter
            gradient_estimate: From neighboring nodes
            phase_alignment: From preserved phase ⟨0.00|0.07⟩
            global_constraint: From C+F=1 enforcement

        Returns:
            reconstructed_value: Final estimate
            contributions: Breakdown by method
        """
        weights = {
            'kalman': 0.40,
            'gradient': 0.20,
            'phase': 0.30,
            'constraint': 0.10
        }

        reconstructed = (
            weights['kalman'] * kalman_prediction +
            weights['gradient'] * gradient_estimate +
            weights['phase'] * phase_alignment +
            weights['constraint'] * global_constraint
        )

        contributions = {
            'kalman': weights['kalman'] * kalman_prediction,
            'gradient': weights['gradient'] * gradient_estimate,
            'phase': weights['phase'] * phase_alignment,
            'constraint': weights['constraint'] * global_constraint,
            'total': reconstructed
        }

        return reconstructed, contributions

    def compute_fidelity(self,
                        reconstructed: np.ndarray,
                        ground_truth: np.ndarray) -> float:
        """
        Compute reconstruction fidelity

        Fidelity = 1 - mean(|reconstructed - ground_truth| / ground_truth)

        Args:
            reconstructed: Reconstructed values
            ground_truth: Actual values

        Returns:
            fidelity: In range [0, 1]
        """
        relative_error = np.abs(reconstructed - ground_truth) / ground_truth
        fidelity = 1.0 - np.mean(relative_error)
        return fidelity


# Example: Simulate chaos test reconstruction
def simulate_chaos_test():
    """Simulate 14 March chaos test with Kalman reconstruction"""

    # Parameters
    gap_start = 400
    gap_end = 600
    gap_duration = gap_end - gap_start
    total_handovers = 1000

    # Generate ground truth syzygy evolution
    t = np.arange(total_handovers)
    true_syzygy = 0.94 + 0.001 * np.sin(2 * np.pi * t / 100) + \
                  0.0001 * np.random.randn(total_handovers)

    # Initialize Kalman filter with observations before gap
    kf = CoherenceKalmanFilter()

    # Train on pre-gap data
    for i in range(gap_start):
        kf.predict()
        kf.update(true_syzygy[i])

    print(f"Kalman gain after training: {kf.get_kalman_gain():.4f}")

    # Predict during gap
    gap_predictions = kf.predict_during_gap(gap_duration)

    # Simulate gradient continuity (interpolation from neighbors)
    gradient_estimates = np.linspace(
        true_syzygy[gap_start-1],
        true_syzygy[gap_end],
        gap_duration
    )

    # Simulate phase alignment (preserved at ~0.94)
    phase_alignment = np.ones(gap_duration) * 0.94

    # Simulate global constraint (C+F=1 enforcement)
    global_constraint = np.ones(gap_duration) * 0.94

    # Reconstruct
    reconstructor = DistributedReconstruction(
        nodes_affected=450,
        nodes_support=12144
    )

    reconstructed = np.zeros(gap_duration)
    for i in range(gap_duration):
        recon, contrib = reconstructor.reconstruct(
            kalman_prediction=gap_predictions[i],
            gradient_estimate=gradient_estimates[i],
            phase_alignment=phase_alignment[i],
            global_constraint=global_constraint[i]
        )
        reconstructed[i] = recon

    # Compute fidelity
    fidelity = reconstructor.compute_fidelity(
        reconstructed,
        true_syzygy[gap_start:gap_end]
    )

    print(f"\nReconstruction Results:")
    print(f"  Gap duration: {gap_duration} handovers")
    print(f"  Nodes affected: {reconstructor.nodes_affected}")
    print(f"  Nodes support: {reconstructor.nodes_support}")
    print(f"  Support ratio: {reconstructor.nodes_support/reconstructor.nodes_affected:.1f}:1")
    print(f"  Fidelity: {fidelity:.6f} ({fidelity*100:.4f}%)")

    # Plot
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Full trajectory
    axes[0].plot(t, true_syzygy, 'b-', alpha=0.5, label='Ground truth')
    axes[0].plot(t[gap_start:gap_end], reconstructed, 'r-',
                linewidth=2, label='Reconstructed')
    axes[0].axvspan(gap_start, gap_end, alpha=0.2, color='gray',
                   label='Gap region')
    axes[0].set_xlabel('Handover')
    axes[0].set_ylabel('Syzygy')
    axes[0].set_title('Chaos Test: Reconstruction During Gap')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Error
    error = np.zeros(total_handovers)
    error[gap_start:gap_end] = reconstructed - true_syzygy[gap_start:gap_end]

    axes[1].plot(t, error * 100, 'r-', linewidth=1)
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1].axvspan(gap_start, gap_end, alpha=0.2, color='gray')
    axes[1].set_xlabel('Handover')
    axes[1].set_ylabel('Error (%)')
    axes[1].set_title('Reconstruction Error')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('chaos_test_reconstruction.png', dpi=150)
    print("\nPlot saved to chaos_test_reconstruction.png")

    return fidelity

if __name__ == "__main__":
    fidelity = simulate_chaos_test()
