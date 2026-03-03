# Arkhe MCP Server - Runbook de Operações

## Deploy Inicial

```bash
# 1. Configurar AWS credentials
aws configure

# 2. Inicializar Terraform
cd terraform/
terraform init
terraform apply

# 3. Deploy aplicações
kubectl apply -f k8s/
```

## Monitoramento Crítico

| Métrica | Alerta | Ação |
|---------|--------|------|
| `arkhe_phi_divergence` > 0.1 | Warning | Verificar sincronização nós |
| `handover_latency_p99` > 5s | Warning | Escalar horizontalmente |

## Rollback de Emergência

```bash
# Reverter para versão anterior
kubectl rollout undo deployment/arkhe-mcp-server -n arkhe-system
```
