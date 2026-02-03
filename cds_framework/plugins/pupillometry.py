# cds_framework/plugins/pupillometry.py
import numpy as np

def predict_attentional_cost(phi_history):
    """
    Implements the 'Killer Prediction':
    Attentional cost (measured by pupil dilation) scales with the square of
    the order parameter change (ΔΦ)².
    """
    # Calculate ΔΦ (change between consecutive steps)
    delta_phi = np.diff(phi_history, prepend=phi_history[0])

    # Cost scales as (ΔΦ)²
    predicted_dilation = delta_phi**2

    # Add a physiological decay/smoothing to simulate pupil response
    # (Typically a slow response, peak ~1s)
    kernel_size = 50
    kernel = np.exp(-np.arange(kernel_size) / 10)
    kernel /= kernel.sum()

    smoothed_dilation = np.convolve(predicted_dilation, kernel, mode='same')

    return smoothed_dilation
