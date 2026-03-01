# Referência da API
Detalhamento de gRPC e REST interfaces.

## 📊 Monitoramento MLflow no Grafana
Para integrar métricas de treinamento/teste de carga:
1. Instale o **Infinity Datasource** no Grafana.
2. Configure a URL: `http://mlflow-service.mlflow:5000`.
3. Use queries REST para `/api/2.0/mlflow/runs/get` filtrando por tags de versão.
