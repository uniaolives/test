"""
BIOGENESIS SERVICE - WebMCP Implementation
Exposes Cognitive Bio-Genesis simulation tools to AI agents via Google Chrome WebMCP.
"""

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
from typing import Dict, List, Optional
import json

from ..core.particle_system import BioGenesisEngine

app = FastAPI(title="Avalon Bio-Genesis WebMCP Service")

# Simulation instance
engine = BioGenesisEngine(num_agents=200)

@app.get("/", response_class=HTMLResponse)
async def index():
    """
    Renders the WebMCP interface for the simulation.
    Uses 'toolname' and 'tooldescription' attributes for agent discovery.
    """
    stats = engine.get_stats()

    html_content = f"""
    <html>
        <head>
            <title>Avalon Bio-Genesis WebMCP</title>
            <style>
                body {{ font-family: sans-serif; background: #121212; color: #e0e0e0; padding: 20px; }}
                .card {{ background: #1e1e1e; border: 1px solid #333; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
                h1 {{ color: #00ffcc; }}
                .stat-value {{ color: #00ccff; font-weight: bold; }}
                input, button {{ padding: 8px; border-radius: 4px; border: 1px solid #444; background: #2a2a2a; color: white; }}
                button {{ background: #0066cc; cursor: pointer; border: none; }}
                button:hover {{ background: #0088ff; }}
                form {{ margin-top: 10px; }}
            </style>
        </head>
        <body>
            <h1>🧬 Bio-Gênese Cognitiva WebMCP</h1>

            <div class="card">
                <h2>Simulation Status</h2>
                <p>Agents: <span class="stat-value">{stats['agents']}</span></p>
                <p>Time: <span class="stat-value">{stats['time']:.2f}</span></p>
                <p>Bonds: <span class="stat-value">{stats['bonds']}</span></p>
                <p>Deaths: <span class="stat-value">{stats['deaths']}</span></p>
            </div>

            <div class="card">
                <h2>Control Tools (WebMCP Enabled)</h2>

                <form action="/inject" method="post" toolname="inject_signal" tooldescription="Inject an energy signal into the morphogenetic field at specific coordinates to attract agents.">
                    <h3>Inject Signal</h3>
                    X: <input type="number" name="x" value="50" step="0.1">
                    Y: <input type="number" name="y" value="50" step="0.1">
                    Z: <input type="number" name="z" value="50" step="0.1">
                    Strength: <input type="number" name="strength" value="20" step="0.1">
                    <button type="submit">Inject</button>
                </form>

                <form action="/restart" method="post" toolname="restart_simulation" tooldescription="Restart the simulation from scratch with initial population.">
                    <h3>System Reset</h3>
                    Agents: <input type="number" name="num_agents" value="200">
                    <button type="submit">Restart Simulation</button>
                </form>
            </div>

            <div class="card">
                <h2>Agent Search</h2>
                <form action="/agent" method="get" toolname="get_agent_info" tooldescription="Get detailed information about a specific agent by its ID.">
                    Agent ID: <input type="number" name="aid" value="0">
                    <button type="submit">Get Details</button>
                </form>
            </div>

            <script>
                // Auto-refresh stats every 5 seconds if not interacting
                setInterval(() => {{
                    // Only refresh if the user isn't typing in an input
                    if (document.activeElement.tagName !== 'INPUT') {{
                        location.reload();
                    }}
                }}, 5000);
            </script>
        </body>
    </html>
    """
    return html_content

@app.post("/inject")
async def inject_signal(x: float = Form(...), y: float = Form(...), z: float = Form(...), strength: float = Form(...)):
    engine.add_signal_source(np.array([x, y, z]), strength, duration=50)
    return {"status": "success", "message": f"Injected signal of strength {strength} at ({x}, {y}, {z})"}

@app.post("/restart")
async def restart(num_agents: int = Form(200)):
    global engine
    engine = BioGenesisEngine(num_agents=num_agents)
    return {"status": "success", "message": f"Simulation restarted with {num_agents} agents."}

@app.get("/agent")
async def get_agent(aid: int):
    info = engine.get_agent_info(aid)
    if info:
        return info
    return {"error": "Agent not found"}

@app.get("/status")
async def get_status():
    return engine.get_stats()

def run_service(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)
