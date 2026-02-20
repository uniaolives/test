#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARKHE(N) 24-Hour HWIL Dashboard with Spectral Visualization
Visualiza:
- Coerência global Φ
- Erro do preditor Kalman
- Dissipação D₂ e D₃
- Espectro de potência da fase com referência k⁻²
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from scipy import signal
import os
from datetime import datetime

# Configurações
CSV_LOW_RATE = "arkhen_24h_telemetry.csv"   # Métricas lentas (coerência, etc.)
CSV_HIGH_RATE = "phase_trace.csv"            # Série temporal de alta frequência (ex.: fase)
FFT_WINDOW_SECONDS = 10                       # Janela de tempo para a FFT (últimos N segundos)
SAMPLE_RATE_HZ = 100                          # Taxa de amostragem esperada (ajustada para 100Hz conforme atmosphere)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("⚛️ Arkhe(N) 24-Hour HWIL Telemetry", style={'textAlign': 'center'}),

    html.Div([
        html.H3("Métricas Globais"),
        dcc.Graph(id='live-coherence'),
        dcc.Graph(id='live-phase-error'),
        dcc.Graph(id='live-dissipation'),
    ]),

    html.Div([
        html.H3("Análise Espectral - Caudas de Momento"),
        html.P("Espectro de potência da fase (últimos {} segundos)".format(FFT_WINDOW_SECONDS)),
        dcc.Graph(id='live-psd'),
    ]),

    dcc.Interval(id='graph-update', interval=2000),  # atualiza a cada 2 segundos
])

def read_low_rate_data():
    """Lê o CSV de métricas lentas e retorna um DataFrame."""
    if not os.path.exists(CSV_LOW_RATE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(CSV_LOW_RATE)
        return df
    except Exception:
        return pd.DataFrame()

def read_high_rate_data():
    """
    Lê o CSV de alta frequência e retorna as últimas amostras
    dentro da janela temporal definida.
    """
    if not os.path.exists(CSV_HIGH_RATE):
        # Gerar dados dummy para visualização se não houver arquivo (opcional)
        return None
    try:
        df = pd.read_csv(CSV_HIGH_RATE)
        if len(df) == 0:
            return None
        # Garantir que há colunas 'timestamp' e 'phase'
        if 'timestamp' not in df.columns or 'phase' not in df.columns:
            return None

        # Converter timestamp para segundos
        df['time_sec'] = df['timestamp'] - df['timestamp'].iloc[0]
        # Selecionar apenas a janela mais recente
        max_time = df['time_sec'].max()
        window_start = max_time - FFT_WINDOW_SECONDS
        df_window = df[df['time_sec'] >= window_start].copy()
        return df_window
    except Exception:
        return None

def compute_psd(df_window, fs=SAMPLE_RATE_HZ):
    """
    Calcula a densidade espectral de potência (método de Welch) da coluna 'phase'.
    Retorna arrays de frequências e PSD, e também a reta teórica k⁻².
    """
    if df_window is None or len(df_window) < 10:
        return None, None, None

    # Extrair sinal e remover tendência linear para evitar vazamento espectral
    signal_data = df_window['phase'].values
    signal_data = signal.detrend(signal_data)

    # Parâmetros da FFT
    nperseg = min(256, len(signal_data))
    if nperseg < 64:
        return None, None, None

    freqs, psd = signal.welch(signal_data, fs=fs, nperseg=nperseg, noverlap=nperseg//2)

    # Gerar reta de referência k^{-2} na mesma escala
    if len(psd) > 0 and freqs[0] > 0:
        # A reta será da forma C * f^{-2}
        C = psd[0] * (freqs[0] ** 2)
        ref_line = C / (freqs ** 2)
    else:
        ref_line = None

    return freqs, psd, ref_line

@app.callback(
    [Output('live-coherence', 'figure'),
     Output('live-phase-error', 'figure'),
     Output('live-dissipation', 'figure'),
     Output('live-psd', 'figure')],
    [Input('graph-update', 'n_intervals')]
)
def update_graphs(n):
    # --- Gráficos de métricas lentas ---
    df_low = read_low_rate_data()

    # Coerência global
    coh_fig = go.Figure()
    if not df_low.empty and 'GlobalCoherence' in df_low.columns:
        coh = go.Scatter(
            x=df_low['Uptime(s)'].tail(600),
            y=df_low['GlobalCoherence'].tail(600),
            mode='lines',
            name='Φ'
        )
        coh_fig.add_trace(coh)
        coh_fig.update_layout(title='Coerência Global', xaxis_title='Tempo (s)',
                              yaxis_title='Φ', yaxis=dict(range=[0.8, 1.02]))
    else:
        coh_fig.update_layout(title='Coerência Global (Aguardando dados...)')

    # Erro de fase
    phase_fig = go.Figure()
    if not df_low.empty and 'PhaseError' in df_low.columns:
        phase = go.Scatter(
            x=df_low['Uptime(s)'].tail(600),
            y=df_low['PhaseError'].tail(600),
            mode='lines',
            name='Erro (rad)'
        )
        phase_fig.add_trace(phase)
        phase_fig.update_layout(title='Erro do Preditor Kalman',
                                xaxis_title='Tempo (s)', yaxis_title='Erro (rad)')
    else:
        phase_fig.update_layout(title='Erro do Preditor (Aguardando dados...)')

    # Dissipação D₂ e D₃
    diss_fig = go.Figure()
    if not df_low.empty and 'D2' in df_low.columns and 'D3' in df_low.columns:
        d2 = go.Scatter(x=df_low['Uptime(s)'].tail(600), y=df_low['D2'].tail(600),
                        mode='lines', name='D₂')
        d3 = go.Scatter(x=df_low['Uptime(s)'].tail(600), y=df_low['D3'].tail(600),
                        mode='lines', name='D₃')
        diss_fig.add_trace(d2)
        diss_fig.add_trace(d3)
        diss_fig.update_layout(title='Dissipação Universal',
                               xaxis_title='Tempo (s)', yaxis_title='Intensidade')
    else:
        diss_fig.update_layout(title='Dissipação (Aguardando dados...)')

    # --- Gráfico espectral (caudas de momento) ---
    psd_fig = go.Figure()
    df_high = read_high_rate_data()
    if df_high is not None:
        freqs, psd, ref_line = compute_psd(df_high, fs=SAMPLE_RATE_HZ)
        if freqs is not None:
            # PSD medida
            psd_trace = go.Scatter(
                x=freqs,
                y=psd,
                mode='lines',
                name='PSD medida',
                line=dict(color='cyan')
            )
            psd_fig.add_trace(psd_trace)

            # Reta de referência k⁻²
            if ref_line is not None:
                ref_trace = go.Scatter(
                    x=freqs,
                    y=ref_line,
                    mode='lines',
                    name='k⁻² (referência)',
                    line=dict(color='red', dash='dash')
                )
                psd_fig.add_trace(ref_trace)

            psd_fig.update_layout(
                title='Espectro de Potência da Fase',
                xaxis_title='Frequência (Hz)',
                yaxis_title='PSD',
                xaxis_type='log',
                yaxis_type='log'
            )
        else:
            psd_fig.update_layout(title='Espectro (poucos dados)')
    else:
        psd_fig.update_layout(title='Aguardando dados de alta frequência...')

    return coh_fig, phase_fig, diss_fig, psd_fig

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)
