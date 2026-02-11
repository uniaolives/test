"""
Servidor MCP (Model Context Protocol) para Arkhe(n) OS.
Inclui suporte avançado para Q-Shield (QEC) e Quantum Search (Grover).
"""

from fastmcp import FastMCP
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("Arkhe.MCP")

def create_mcp_server(arkhe_system, parallax_client=None):
    mcp = FastMCP("ArkheOS")

    @mcp.tool()
    def get_system_status() -> Dict[str, Any]:
        if arkhe_system.simulation:
            stats = arkhe_system.simulation.get_stats()
            return {
                "status": "operational",
                "simulation": stats,
                "field_active": arkhe_system.field is not None,
                "distributed": parallax_client is not None and parallax_client.running,
                "q_shield_active": parallax_client.qec_active if parallax_client else False
            }
        return {"status": "offline"}

    @mcp.tool()
    async def activate_q_shield(status: bool) -> str:
        """Ativa/Desativa correção de erros quânticos (Surface Code)"""
        if parallax_client:
            parallax_client.qec_active = status
            return f"🛡️ Q-Shield {'ATIVADO' if status else 'DESATIVADO'}"
        return "Erro: Parallax não disponível"

    @mcp.tool()
    async def quantum_search(genome_signature: int) -> Dict:
        """Busca agente por assinatura de genoma usando Grover Distribuído O(√N)"""
        if not arkhe_system.grover:
            return {"error": "Grover Engine não inicializada"}

        result = await arkhe_system.grover.run_search(genome_signature)
        return {"status": "COMPLETED", "result": result}

    @mcp.tool()
    async def quantum_entangle_global(agent_a_id: int, agent_b_node: str, agent_b_id: int, bell_type: int = 0) -> Dict:
        if not parallax_client: return {"error": "No Parallax"}
        success = await parallax_client.entangle_remote(agent_a_id, agent_b_node, agent_b_id, bell_type)
        return {"success": success}

    @mcp.tool()
    def inject_field_signal(x: float, y: float, z: float, strength: float) -> str:
        if arkhe_system.simulation:
            arkhe_system.simulation.inject_signal(x, y, z, strength)
            return f"Sinal injetado"
        return "Erro"

    return mcp
