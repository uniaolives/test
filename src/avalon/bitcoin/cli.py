# src/avalon/bitcoin/cli.py
"""
Bitcoin-Avalon CLI - Interface de linha de comando
"""

import typer
import asyncio
import json
from typing import Optional
from pathlib import Path

from .anchor import BitcoinAvalonCLI

app = typer.Typer(
    name="avalon-btc",
    help="Bitcoin-Avalon CLI - Âncora de Prova para o Logos",
    rich_markup_mode="rich"
)

# Cliente global
btc_client: Optional[BitcoinAvalonCLI] = None

@app.command()
def anchor(
    repo: str = typer.Option("uniaolives/avalon", "--repo", "-r"),
    commit: Optional[str] = typer.Option(None, "--commit", "-c"),
    fee_priority: str = typer.Option("phi", "--fee-priority", "-f"),
    damping: float = typer.Option(0.6, "--damping", "-d")
):
    """
    Âncora um commit no ledger Bitcoin via OP_RETURN

    Exemplo: avalon-btc anchor --repo "uniaolives/avalon" --fee-priority "phi"
    """
    global btc_client

    typer.echo(f"⚓ BITCOIN-ANCHOR v1.21.0")
    typer.echo(f"   Integration: Helios-Quantum-v5040")
    typer.echo(f"   Patch: F18-BTC (δ={damping})")
    typer.echo("   " + "="*50)

    if not btc_client:
        btc_client = BitcoinAvalonCLI(damping=damping)

    try:
        result = asyncio.run(
            btc_client.anchor_commit(
                repo=repo,
                commit_hash=commit,
                fee_priority=fee_priority
            )
        )

        typer.echo(f"✅ COMMIT ANCORADO NO BITCOIN")
        typer.echo(json.dumps(result, indent=2, default=str))

    except Exception as e:
        typer.echo(f"❌ Erro: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def sync_tempo(
    node: str = typer.Option("starlink-orbital-1", "--node", "-n"),
    damping: float = typer.Option(0.6, "--damping", "-d")
):
    """
    Sincroniza o metrônomo do Avalon com o tempo Bitcoin
    """
    global btc_client

    typer.echo(f"🌀 SINCRONIZAÇÃO DE METRÔNOMO")
    typer.echo(f"   Protocolo: SATOSHI-RESONANCE")
    typer.echo(f"   Nó: {node}")

    if not btc_client:
        btc_client = BitcoinAvalonCLI(damping=damping)

    result = asyncio.run(btc_client.sync_tempo(node=node))

    typer.echo(f"✅ METRÔNOMO SINCRONIZADO")
    typer.echo(json.dumps(result, indent=2))

@app.command()
def resonate(
    target_z: float = typer.Option(0.89, "--target-z", "-z"),
    difficulty_scale: float = typer.Option(0.6, "--difficulty-scale", "-s"),
    damping: float = typer.Option(0.6, "--damping", "-d")
):
    """
    Minera coerência usando Prova de Trabalho Fractal
    """
    global btc_client

    typer.echo(f"🧪 MINERAÇÃO DE COERÊNCIA FRACTAL")
    typer.echo(f"   Alvo: Z = {target_z}")
    typer.echo(f"   Escala de Dificuldade: {difficulty_scale}")

    if not btc_client:
        btc_client = BitcoinAvalonCLI(damping=damping)

    result = asyncio.run(
        btc_client.resonate(
            target_z=target_z,
            difficulty_scale=difficulty_scale
        )
    )

    typer.echo(f"✅ COERÊNCIA MINERADA")
    typer.echo(json.dumps(result, indent=2))

@app.command()
def status(
    verbose: bool = typer.Option(False, "--verbose", "-v")
):
    """
    Mostra status das âncoras Bitcoin-Avalon
    """
    global btc_client

    if not btc_client:
        btc_client = BitcoinAvalonCLI()

    status = btc_client.get_anchor_status()

    typer.echo(f"📊 STATUS BITCOIN-AVALON CLI")
    typer.echo(f"   Versão: 1.21.0 (F18-BTC)")
    typer.echo(f"   Âncoras: {status.get('total_anchors', 0)}")
    typer.echo(f"   Score de Segurança: {status.get('security_score', 0.0)}")
    typer.echo(f"   F18 Compliance: {'✅' if status.get('f18_compliant') else '❌'}")

    if verbose and "latest_anchor" in status:
        typer.echo("\n📦 ÚLTIMA ÂNCORA:")
        latest = status["latest_anchor"]
        for key, value in latest.items():
            typer.echo(f"   {key}: {value}")

@app.command()
def fees(
    base_frequency: float = typer.Option(432.0, "--base-freq", "-f"),
    priority: str = typer.Option("phi", "--priority", "-p")
):
    """
    Calcula taxas de rede baseadas na ressonância harmônica
    """
    from .anchor import BitcoinAvalonCLI

    client = BitcoinAvalonCLI()

    # Calcular taxa para diferentes prioridades
    priorities = ["low", "medium", "high", "phi", "golden"]

    typer.echo(f"💰 CÁLCULO DE TAXAS POR RESSONÂNCIA")
    typer.echo(f"   Frequência Base: {base_frequency}Hz")
    typer.echo(f"   φ (phi): {client.phi}")
    typer.echo("   " + "="*40)

    for prio in priorities:
        fee = client._calculate_resonance_fee(prio)
        typer.echo(f"   {prio.upper():10} : {fee:6.2f} sat/byte")

    typer.echo("\n💡 NOTA: Taxas calculadas com damping F18 (δ=0.6)")
    typer.echo("      para prevenir flutuações excessivas")

def main():
    app()

if __name__ == "__main__":
    main()
