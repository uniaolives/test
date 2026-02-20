#!/usr/bin/env python3
# arkhen_dashboard.py
# Dashboard em tempo real para a maratona HWIL de 24h

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import time
import os

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("⚛️ Arkhe(N) 24-Hour HWIL Telemetry"),
    dcc.Graph(id='live-phase-graph'),
    dcc.Graph(id='live-coherence-graph'),
    dcc.Interval(id='graph-update', interval=2000)  # atualiza a cada 2s
])

@app.callback(
    [Output('live-phase-graph', 'figure'),
     Output('live-coherence-graph', 'figure')],
    [Input('graph-update', 'n_intervals')]
)
def update_graphs(n):
    if not os.path.exists('arkhen_24h_telemetry.csv'):
        return go.Figure(), go.Figure()

    try:
        # Lê o arquivo CSV gerado pelo orquestrador
        df = pd.read_csv('arkhen_24h_telemetry.csv')
    except:
        # Se houver erro na leitura (ex: arquivo sendo escrito)
        return go.Figure(), go.Figure()

    if df.empty:
        return go.Figure(), go.Figure()

    # Gráfico da fase (últimos 600 pontos)
    phase_trace = go.Scatter(
        x=df['Uptime(s)'].tail(600),
        y=df['PhaseError'].tail(600),
        mode='lines',
        name='Erro de Fase (rad)'
    )
    phase_layout = go.Layout(title='Erro de Fase do Preditor Kalman',
                              xaxis=dict(title='Tempo de Teste (s)'),
                              yaxis=dict(title='Erro (rad)'))

    # Gráfico da coerência global
    coh_trace = go.Scatter(
        x=df['Uptime(s)'].tail(600),
        y=df['GlobalCoherence'].tail(600),
        mode='lines+markers',
        name='Φ global'
    )
    coh_layout = go.Layout(title='Coerência Global',
                            xaxis=dict(title='Tempo de Teste (s)'),
                            yaxis=dict(title='Φ', range=[0.8, 1.02]))

    return go.Figure(data=[phase_trace], layout=phase_layout), \
           go.Figure(data=[coh_trace], layout=coh_layout)

if __name__ == '__main__':
    # Usar porta alternativa para evitar conflitos
    app.run(debug=False, host='0.0.0.0', port=8050)
