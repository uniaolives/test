"""
Core visualization engine for the Temporal Weather Map dashboard.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ChronofluxDashboard:
    """
    Generates the visualization for the public-facing Temporal Weather Map.
    """

    def __init__(self, node_data, map_center=None):
        self.node_data = node_data
        self.map_center = map_center or {'lat': -15, 'lon': -60}

    def create_global_map(self, current_time):
        """
        Create the main global map visualization with node indicators.
        """
        fig = go.Figure()

        # Add base world map
        fig.add_trace(go.Scattergeo(
            lon=[-180, 180, 180, -180, -180],
            lat=[-90, -90, 90, 90, -90],
            mode='lines',
            line=dict(width=0.5, color='gray'),
            showlegend=False,
            hoverinfo='skip'
        ))

        node_lons = []
        node_lats = []
        node_text = []
        node_sizes = []
        node_colors = []

        for node_id, data in self.node_data.items():
            node_lons.append(data['longitude'])
            node_lats.append(data['latitude'])

            text = (f"<b>Node {node_id}</b><br>"
                   f"Location: {data.get('city', 'Unknown')}<br>"
                   f"Iχ Index: {data.get('Iχ', 0):.2f}<br>"
                   f"Vorticity: {data.get('omega', 0):.3e}<br>")
            node_text.append(text)

            node_sizes.append(10 + data.get('Iχ', 0) * 5)

            Iχ = data.get('Iχ', 0)
            if Iχ < 3:
                color = 'blue'
            elif Iχ < 6:
                color = 'yellow'
            else:
                color = 'red'
            node_colors.append(color)

        fig.add_trace(go.Scattergeo(
            lon=node_lons,
            lat=node_lats,
            text=node_text,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color='white'),
                opacity=0.8
            ),
            name='T.E.N.S.O.R. Nodes',
            hoverinfo='text'
        ))

        fig.update_layout(
            title=f"Earth's Temporal Weather Map - {current_time}",
            geo=dict(
                projection_type='natural earth',
                showland=True,
                center=self.map_center,
            ),
            height=600
        )

        return fig

    def create_time_series_plot(self, node_id, history_data):
        """
        Create time series plot for a specific node.
        """
        if node_id not in history_data:
            return None

        data = history_data[node_id]
        times = data['timestamps']
        Iχ_values = data['Iχ_values']
        omega_values = data['omega_values']

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Iχ Index Over Time', 'Temporal Vorticity ω'),
            vertical_spacing=0.15
        )

        fig.add_trace(
            go.Scatter(x=times, y=Iχ_values, mode='lines+markers', name='Iχ Index'),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=times, y=omega_values, mode='lines+markers', name='ω'),
            row=2, col=1
        )

        fig.update_layout(height=500, title=f"Node {node_id} History")
        return fig
