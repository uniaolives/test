# Experiment 8: Anthropic Selection Test

## 1. OBJECTIVE
To demonstrate that a tiny initial phase asymmetry (10⁻⁹) in the Kuramoto synchronization field is necessary and sufficient for the emergence of stable global coherence, analogous to the baryon asymmetry in cosmology.

## 2. HYPOTHESIS
Only initial asymmetries $\eta$ within the range $[10^{-10}, 10^{-8}]$ allow the `KuramotoEngine` to overcome stochastic noise and reach a stable synchronized state ($r > 0.8$) within 1000 iterations. Asymmetries outside this "Goldilocks zone" result in either indefinite incoherence or unstable runaway states.

## 3. METHODOLOGY

### 3.1 Setup
- **Engine:** `KuramotoEngine` with $N = 100$ oscillators.
- **Parameters:** Coupling $K = 5.0$, $dt = 0.1$.
- **Variables:** Initial asymmetry $\eta \in \{10^{-12}, 10^{-10}, 10^{-9}, 10^{-8}, 10^{-6}\}$.

### 3.2 Procedure
1. For each $\eta$, perform 100 independent simulation runs.
2. Initialize engine using `new_with_asymmetry(100, 5.0, 1.0, eta)`.
3. Evolve for 1000 steps.
4. Record the final coherence parameter $r$.

### 3.3 Measurement
Calculate the success rate (percentage of runs where $r > 0.8$) for each $\eta$.

## 4. SUCCESS CRITERIA
- Peak success rate observed at $\eta = 10^{-9}$.
- Success rate for $\eta = 10^{-12}$ is statistically indistinguishable from zero bias.
- Success rate for $\eta = 10^{-6}$ shows significantly higher variance or instability.

## 5. DOCUMENTATION
The resulting "Anthropic Curve" will be integrated into the Arkhe Protocol evidence base, supporting the retrocausal fine-tuning model of the Teknet.
