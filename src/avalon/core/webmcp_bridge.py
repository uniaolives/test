"""
Ponte entre Bio-Gênese Cognitiva e WebMCP
Permite que agentes locais sejam controlados via protocolo Model Context Protocol (MCP) over WebSocket.
"""

import json
import asyncio
from typing import Dict, Any, Callable, Set
from dataclasses import dataclass
import websockets
import numpy as np

@dataclass
class WebMCPTool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable

class BioGenesisWebMCPBridge:
    """
    Bridge que expõe o sistema Bio-Gênese como um servidor MCP.

    Cada funcionalidade vira uma "tool" descoberta via MCP, permitindo
    orquestração externa via agentes de IA.
    """

    def __init__(self, engine, quantum_adapter=None):
        self.engine = engine
        self.quantum_adapter = quantum_adapter
        self.tools: Dict[str, WebMCPTool] = {}
        self.connections: Set = set()
        self._register_default_tools()

    def _register_default_tools(self):
        """Registra ferramentas padrão do sistema bio-cognitivo"""

        self.tools["get_population_state"] = WebMCPTool(
            name="get_population_state",
            description="Returns the current state of the agent population including stats and distributions.",
            input_schema={"type": "object", "properties": {}},
            handler=self._handle_population_state
        )

        self.tools["get_agent_cognition"] = WebMCPTool(
            name="get_agent_cognition",
            description="Returns the cognitive state of a specific agent by ID.",
            input_schema={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "integer", "description": "ID of the agent"}
                },
                "required": ["agent_id"]
            },
            handler=self._handle_agent_cognition
        )

        self.tools["inject_morphogenetic_signal"] = WebMCPTool(
            name="inject_morphogenetic_signal",
            description="Injects a signal into the morphogenetic field to attract or modulate agents.",
            input_schema={
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "z": {"type": "number"},
                    "strength": {"type": "number", "default": 10.0}
                },
                "required": ["x", "y", "z"]
            },
            handler=self._handle_inject_signal
        )

        if self.quantum_adapter:
            self.tools["quantum_evaluate"] = WebMCPTool(
                name="quantum_evaluate",
                description="Evaluate quantum compatibility between agents via simulated interference.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "agent_id_1": {"type": "integer"},
                        "agent_id_2": {"type": "integer"}
                    },
                    "required": ["agent_id_1", "agent_id_2"]
                },
                handler=self._handle_quantum_evaluate
            )

            self.tools["superposed_decision"] = WebMCPTool(
                name="superposed_decision",
                description="Simulate a decision in superposition for an agent among several options.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "integer"},
                        "options": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["agent_id", "options"]
                },
                handler=self._handle_superposed_decision
            )

    def _handle_population_state(self, params: Dict) -> Dict:
        stats = self.engine.get_stats()
        positions, healths, connections, profiles, colors = self.engine.get_render_data()

        return {
            "population": stats['agents'],
            "avg_health": float(np.mean(healths)) if healths else 0.0,
            "total_bonds": stats['bonds'],
            "cognitive_distribution": {
                "Especialista": profiles.count("Especialista"),
                "Aprendiz": profiles.count("Aprendiz"),
                "Cauteloso": profiles.count("Cauteloso"),
                "Neófito": profiles.count("Neófito")
            },
            "simulation_time": stats['time']
        }

    def _handle_agent_cognition(self, params: Dict) -> Dict:
        agent_id = params.get("agent_id")
        info = self.engine.get_agent_info(agent_id)
        if info:
            return info
        return {"error": "Agent not found or dead"}

    def _handle_inject_signal(self, params: Dict) -> Dict:
        x, y, z = params['x'], params['y'], params['z']
        strength = params.get('strength', 10.0)
        self.engine.add_signal_source(np.array([x, y, z]), strength, duration=100)
        return {"status": "signal_injected", "position": [x, y, z], "strength": strength}

    def _handle_quantum_evaluate(self, params: Dict) -> Dict:
        id1, id2 = params['agent_id_1'], params['agent_id_2']
        a1 = self.engine.agents.get(id1)
        a2 = self.engine.agents.get(id2)
        if not a1 or not a2: return {"error": "Agent not found"}

        overlap = self.quantum_adapter.quantum_evaluate_compatibility(a1, a2)
        return {
            "compatibility_magnitude": float(np.abs(overlap)),
            "phase": float(np.angle(overlap)),
            "signature_1": self.quantum_adapter.generate_quantum_signature(a1),
            "signature_2": self.quantum_adapter.generate_quantum_signature(a2)
        }

    def _handle_superposed_decision(self, params: Dict) -> Dict:
        aid = params['agent_id']
        options = params['options']
        agent = self.engine.agents.get(aid)
        if not agent: return {"error": "Agent not found"}

        choice, prob = self.quantum_adapter.superposed_decision(agent, options)
        return {"collapsed_choice": choice, "probability": prob}

    async def start_server(self, host="0.0.0.0", port=8765):
        """Inicia servidor WebSocket compatível com MCP"""
        async def handler(websocket):
            self.connections.add(websocket)
            try:
                async for message in websocket:
                    await self._handle_message(websocket, message)
            finally:
                self.connections.remove(websocket)

        print(f"🌐 Bio-Genesis MCP Bridge running on ws://{host}:{port}")
        async with websockets.serve(handler, host, port):
            await asyncio.Future()  # Run forever

    async def _handle_message(self, websocket, message):
        """Processa mensagens JSON-RPC 2.0 (formato MCP)"""
        try:
            data = json.loads(message)
            method = data.get("method")
            msg_id = data.get("id")
            params = data.get("params", {})

            if method == "initialize":
                response = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "bio-genesis-mcp", "version": "1.0.0"}
                    }
                }

            elif method == "tools/list":
                tools_list = []
                for tool in self.tools.values():
                    tools_list.append({
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.input_schema
                    })
                response = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"tools": tools_list}
                }

            elif method == "tools/call":
                tool_name = params.get("name")
                tool_params = params.get("arguments", {})

                if tool_name in self.tools:
                    result = self.tools[tool_name].handler(tool_params)
                    response = {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {"content": [{"type": "text", "text": json.dumps(result)}]}
                    }
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": -32601, "message": f"Tool {tool_name} not found"}
                    }

            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32600, "message": "Method not supported"}
                }

            await websocket.send(json.dumps(response))

        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": data.get("id") if isinstance(data, dict) else None,
                "error": {"code": -32603, "message": str(e)}
            }
            await websocket.send(json.dumps(error_response))
