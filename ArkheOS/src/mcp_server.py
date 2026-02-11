"""
Servidor MCP (Model Context Protocol) para Arkhe(n) OS.
Utiliza FastMCP para facilidade de integração e suporte SSE.
"""

from fastmcp import FastMCP
from typing import Dict, Any

def create_mcp_server(arkhe_system):
    mcp = FastMCP("ArkheOS")

    @mcp.tool()
    def get_system_status() -> Dict[str, Any]:
        """Retorna status do sistema Arkhe(n)"""
        if arkhe_system.simulation:
            stats = arkhe_system.simulation.get_stats()
            return {
                "status": "operational",
                "simulation": stats,
                "field_active": arkhe_system.field is not None
            }
        return {"status": "simulation_not_available"}

    @mcp.tool()
    def inject_field_signal(x: float, y: float, z: float, strength: float) -> str:
        """Injeta sinal no campo morfogenético"""
        if arkhe_system.simulation:
            arkhe_system.simulation.inject_signal(x, y, z, strength)
            return f"Sinal de força {strength} injetado em ({x}, {y}, {z})"
        return "Erro: Simulação não disponível"

    @mcp.tool()
    def query_agent(agent_id: int) -> Dict[str, Any]:
        """Consulta informações de um agente"""
        if arkhe_system.simulation:
            info = arkhe_system.simulation.get_agent_info(agent_id)
            return info if info else {"error": "Agente não encontrado"}
        return {"error": "Simulação não disponível"}

    return mcp
