# interface/human_cli.py
import click
import requests
import json
import os

@click.group()
def cli():
    """SASC Distributed Agent Network - Interface Humana"""
    pass

@cli.command()
@click.option('--strategy', '-s', required=True, help='Arquivo de estratégia JSON')
def deploy(strategy):
    """Deploya nova estratégia na rede"""
    gateway = os.environ.get("VPS_GATEWAY", "vps-gateway")
    with open(strategy) as f:
        strat = json.load(f)

    # Enviar ao Brain
    try:
        # Note: SSL verification is enabled by default for security.
        # Use VPS_CA_BUNDLE env var to specify a custom CA if needed.
        verify_ssl = os.environ.get("SASC_VERIFY_SSL", "True").lower() == "true"
        response = requests.post(
            f"https://{gateway}/api/v1/strategy",
            json=strat,
            headers={"Authorization": f"Bearer {get_token()}"},
            verify=verify_ssl
        )

        if response.status_code == 200:
            click.echo("✅ Estratégia deployada com sucesso")
            click.echo(f"   Nodes afetados: {response.json()['nodes_updated']}")
        else:
            click.echo(f"❌ Erro: {response.text}")
    except Exception as e:
        click.echo(f"❌ Connection error: {e}")

@cli.command()
def status():
    """Status da rede distribuída"""
    gateway = os.environ.get("VPS_GATEWAY", "vps-gateway")
    try:
        verify_ssl = os.environ.get("SASC_VERIFY_SSL", "True").lower() == "true"
        response = requests.get(
            f"https://{gateway}/api/v1/status",
            headers={"Authorization": f"Bearer {get_token()}"},
            verify=verify_ssl
        )

        data = response.json()

        click.echo("🏛️ SASC Distributed Network Status")
        click.echo("=" * 40)
        click.echo(f"Brain: {'🟢 Online' if data['brain_online'] else '🔴 Offline'}")
        click.echo(f"Nodes ativos: {data['active_nodes']}/{data['total_nodes']}")
        click.echo(f"Tasks em execução: {data['running_tasks']}")
        click.echo(f"Load médio: {data['avg_load']:.1%}")

        click.echo("\nNodes:")
        for node in data['nodes']:
            status_icon = "🟢" if node['online'] else "🔴"
            click.echo(f"  {status_icon} {node['id']} ({node['type']}) - {node['load']:.0%}")
    except Exception as e:
        click.echo(f"❌ Connection error: {e}")

@cli.command()
@click.argument('task_type')
@click.option('--data', '-d', help='Dados da tarefa (JSON)')
def task(task_type, data):
    """Submete tarefa à rede"""
    gateway = os.environ.get("VPS_GATEWAY", "vps-gateway")
    payload = {
        'type': task_type,
        'data': json.loads(data) if data else {},
        'priority': 'normal',
        'requires_approval': True  # Human-in-the-loop
    }

    try:
        verify_ssl = os.environ.get("SASC_VERIFY_SSL", "True").lower() == "true"
        response = requests.post(
            f"https://{gateway}/api/v1/tasks",
            json=payload,
            headers={"Authorization": f"Bearer {get_token()}"},
            verify=verify_ssl
        )

        task_id = response.json()['task_id']
        click.echo(f"📋 Tarefa criada: {task_id}")
        click.echo("⏳ Aguardando aprovação humana..." if payload['requires_approval'] else "🚀 Executando...")
    except Exception as e:
        click.echo(f"❌ Connection error: {e}")

def get_token():
    """Obtém token de autenticação"""
    # Implementar OAuth2/JWT
    return os.environ.get("HUMAN_TOKEN", "human_token_here")

if __name__ == '__main__':
    cli()
