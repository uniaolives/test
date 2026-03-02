"""
intelligent_alerting.py
Sistema de alertas baseado em anomalias
"""

try:
    from sklearn.ensemble import IsolationForest
    from prophet import Prophet
except ImportError:
    IsolationForest = None
    Prophet = None
import numpy as np
from datetime import datetime
from typing import Dict, List
import pandas as pd

class IntelligentAlertSystem:
    """Sistema de alertas que aprende padrões normais"""

    def __init__(self):
        # Modelos de detecção de anomalias
        if IsolationForest:
            self.isolation_forest = IsolationForest(
                contamination=0.01,
                random_state=42
            )
        else:
            self.isolation_forest = None

        # Modelo de forecasting
        if Prophet:
            self.prophet_model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_mode='multiplicative'
            )
        else:
            self.prophet_model = None

        # Histórico de métricas
        self.metric_history = {
            'world_model_loss': [],
            'physics_coherence': [],
            'entropy': [],
            'phi': [],
        }

        # Limiares adaptativos
        self.adaptive_thresholds = {}

    def analyze_metrics(self, current_metrics: Dict):
        """Analisa métricas atuais e gera alertas se necessário"""

        alerts = []

        for metric_name, value in current_metrics.items():
            if metric_name not in self.metric_history:
                self.metric_history[metric_name] = []

            # Adiciona ao histórico
            self.metric_history[metric_name].append(value)

            # Mantém histórico limitado
            if len(self.metric_history[metric_name]) > 1000:
                self.metric_history[metric_name] = self.metric_history[metric_name][-1000:]

            # Verifica anomalias
            if self._is_anomalous(metric_name, value):
                alerts.append({
                    'metric': metric_name,
                    'value': value,
                    'threshold': self.adaptive_thresholds.get(metric_name, 'adaptive'),
                    'severity': self._calculate_severity(metric_name, value),
                    'timestamp': datetime.now().isoformat(),
                })

            # Verifica tendências
            trend = self._analyze_trend(metric_name)
            if trend['direction'] == 'worsening' and trend['confidence'] > 0.8:
                alerts.append({
                    'type': 'trend_alert',
                    'metric': metric_name,
                    'trend': trend,
                    'severity': 'warning',
                    'timestamp': datetime.now().isoformat(),
                })

        return alerts

    def _is_anomalous(self, metric_name: str, value: float) -> bool:
        """Detecta se valor é anômalo usando múltiplos métodos"""

        history = self.metric_history[metric_name]

        if len(history) < 100:  # Não tem dados suficientes
            return False

        # 1. Detecção por Isolation Forest
        if self.isolation_forest:
            X = np.array(history).reshape(-1, 1)
            self.isolation_forest.fit(X)
            prediction = self.isolation_forest.predict([[value]])

            if prediction[0] == -1:  # Anomalia
                return True

        # 2. Z-score adaptativo
        mean = np.mean(history)
        std = np.std(history)

        if std > 0:
            z_score = abs((value - mean) / std)
            if z_score > 3.5:  # 3.5 sigma
                return True

        # 3. Forecasting com Prophet
        if self.prophet_model and len(history) > 365:  # Tem dados suficientes para forecasting
            forecast = self._prophet_forecast(metric_name, history)

            if value < forecast['yhat_lower'] or value > forecast['yhat_upper']:
                return True

        return False

    def _prophet_forecast(self, metric_name: str, history: List[float]):
        """Forecast usando Prophet"""

        # Prepara dados para Prophet
        df = pd.DataFrame({
            'ds': pd.date_range(start='2024-01-01', periods=len(history), freq='H'),
            'y': history
        })

        # Treina modelo
        model = Prophet()
        model.fit(df)

        # Faz forecast
        future = model.make_future_dataframe(periods=1, freq='H')
        forecast = model.predict(future)

        return forecast.iloc[-1]

    def _analyze_trend(self, metric_name: str) -> Dict:
        """Analisa tendência da métrica"""

        history = self.metric_history[metric_name]

        if len(history) < 50:
            return {'direction': 'unknown', 'confidence': 0.0}

        # Regressão linear simples
        x = np.arange(len(history))
        y = np.array(history)

        slope, intercept = np.polyfit(x, y, 1)

        # Calcula confiança (R²)
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return {
            'direction': 'improving' if slope < 0 else 'worsening',
            'slope': slope,
            'confidence': r_squared,
            'current_value': history[-1],
        }

    def _calculate_severity(self, name, value): return "warning"
