"""
Servidor MCP (Model Context Protocol) para Arkhe(n) OS.
Interface para o mundo externo com suporte a Bio-Gênese e Quantum.
"""

from fastmcp import FastMCP
from typing import Dict, Any, Optional
import ctypes
import logging

logger = logging.getLogger("Arkhe.MCP")

def create_mcp_server(arkhe_system, parallax_client=None):
    mcp = FastMCP("ArkheOS")

    @mcp.tool()
    def get_system_status() -> Dict[str, Any]:
        """Retorna status vital do sistema Arkhe(n)"""
        if arkhe_system.simulation:
            stats = arkhe_system.simulation.get_stats()
            return {
                "status": "operational",
                "simulation": stats,
                "field_active": arkhe_system.field is not None,
                "distributed": parallax_client is not None and parallax_client.running,
                "quantum_ready": parallax_client is not None and parallax_client.qhttp_dist_lib is not None
            }
        return {"status": "offline"}

    @mcp.tool()
    async def quantum_entangle_global(agent_a_id: int, agent_b_node: str, agent_b_id: int, bell_type: int = 0) -> Dict:
        """Emaranha um agente local com um remoto via cluster Parallax"""
        if not parallax_client:
            return {"error": "Parallax Node Client not available"}

        success = await parallax_client.entangle_remote(
            local_agent=agent_a_id, remote_node=agent_b_node, remote_agent=agent_b_id, bell_type=bell_type
        )

        return {
            "success": success,
            "local_agent": agent_a_id,
            "remote_node": agent_b_node,
            "remote_agent": agent_b_id,
            "bell_type": bell_type
        }

    @mcp.tool()
    async def quantum_collapse_global(agent_id: int) -> Dict:
        """Colapsa o estado quântico de um agente e aplica efeitos Hebbianos"""
        if not parallax_client:
            return {"error": "Parallax not active"}

        measured = await parallax_client.collapse_remote(agent_id)
        if measured is None:
            return {"error": "Quantum collapse failed"}

        # Efeito Hebbiano na simulação local
        effect = "none"
        if arkhe_system.simulation:
            agent = arkhe_system.simulation.agents.get(agent_id)
            if agent:
                # Bases pares = boost, ímpares = penalty
                if measured % 2 == 0:
                    agent.health = min(1.0, agent.health + 0.05)
                    effect = "health_boost"
                else:
                    agent.health = max(0.0, agent.health - 0.03)
                    effect = "health_penalty"

        return {
            "agent_id": agent_id,
            "measured_state": measured,
            "hebbian_effect": effect,
            "binary_result": format(measured, '04b')
        }

    @mcp.tool()
    def inject_field_signal(x: float, y: float, z: float, strength: float) -> str:
        if arkhe_system.simulation:
            arkhe_system.simulation.inject_signal(x, y, z, strength)
            return f"Sinal injetado em ({x},{y},{z})"
        return "Erro: Simulação não disponível"

    @mcp.tool()
    def query_agent(agent_id: int) -> Dict[str, Any]:
        if arkhe_system.simulation:
            info = arkhe_system.simulation.get_agent_info(agent_id)
            return info if info else {"error": "Agente não encontrado"}
        return {"error": "Simulação não disponível"}

    return mcp
