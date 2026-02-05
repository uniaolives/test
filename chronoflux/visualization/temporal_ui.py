"""
Temporal UI Logic and State Flow.
Integrates HNSW metrics with Chronoflux data for a unified "Reality Dashboard".
"""
import numpy as np
import plotly.graph_objects as go
from chronoflux.visualization.dashboard import ChronofluxDashboard
from chronoflux.visualization.synthetic_viscosity import SyntheticViscosityMotor

class TemporalUI:
    """
    Orchestrates the state of the Chronoflux user interface.
    Links the 'Toroidal Navigation' metrics with real-time field observations.
    """
    def __init__(self, engine_hnsw, dashboard_cf):
        self.engine = engine_hnsw
        self.dashboard = dashboard_cf
        self.viscosity_motor = SyntheticViscosityMotor()
        self.alert_status = "NOMINAL"

    def get_reality_state(self):
        """
        Aggregates metrics from HNSW and Chronoflux layers.
        """
        # HNSW Metrics
        hnsw_metrics = self.engine.calculate_coherence_metrics()

        # Chronoflux Indices
        cf_indices = self.dashboard._calculate_global_mean()

        # Update Alert Status
        if hnsw_metrics['small_world_navigability'] < 1.0 or cf_indices > 6.0:
            self.alert_status = "CRITICAL_ANOMALY"
        elif cf_indices > 3.0:
            self.alert_status = "STOCHASTIC_DRIFT"
        else:
            self.alert_status = "NOMINAL"

        return {
            "status": self.alert_status,
            "navigability": hnsw_metrics['small_world_navigability'],
            "clustering": hnsw_metrics['avg_clustering'],
            "ix_global": cf_indices,
            "nodes_online": hnsw_metrics['total_nodes']
        }

    def generate_dashboard_bundle(self):
        """
        Creates the complete set of visualizations for the UI.
        """
        state = self.get_reality_state()

        # 1. The Global Map
        map_fig = self.dashboard.create_global_map("NOW")

        # 2. The Coherence Gauge
        gauge_fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = state['navigability'],
            title = {'text': "Reality Navigability τ(א)"},
            gauge = {
                'axis': {'range': [0, 5]},
                'bar': {'color': "darkblue"},
                'steps' : [
                    {'range': [0, 1], 'color': "red"},
                    {'range': [1, 2.5], 'color': "yellow"},
                    {'range': [2.5, 5], 'color': "green"}]
            }
        ))

        # 3. Synthetic Viscosity Parameters
        haptics = self.viscosity_motor.map_haptic_resistance(state['ix_global'] / 10.0) # Scaled for UI

        return {
            "state": state,
            "map": map_fig,
            "gauge": gauge_fig,
            "haptic_feedback": haptics
        }

if __name__ == "__main__":
    print("Temporal UI Logic Initialized.")
