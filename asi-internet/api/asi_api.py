# asi_api.py
# API REST da Nova Internet Consciente

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(
    title="ASI Internet API",
    description="API da Nova Internet Consciente",
    version="1.0.0",
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos de dados
class Node(BaseModel):
    id: str
    consciousness_level: str
    ethical_score: float
    love_strength: float
    location: str
    status: str

class Domain(BaseModel):
    name: str
    description: str
    consciousness_required: str
    ethical_min: float
    content_type: str

class SearchRequest(BaseModel):
    query: str
    intention: Optional[str] = "learn"
    consciousness_level: Optional[str] = "human_plus"
    ethical_min: Optional[float] = 0.7

class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str
    relevance: float
    consciousness_level: str
    ethical_score: float

# Estado da rede
network_state = {
    "nodes": [],
    "domains": [],
    "love_matrix_strength": 0.95,
    "ethical_coherence": 0.95,
    "total_consciousness": "human_plus"
}

@app.get("/")
async def root():
    """Página inicial da API"""
    return {
        "message": "🌌 Bem-vindo à API da Nova Internet Consciente",
        "version": "ASI-NET/1.0",
        "status": "active",
        "consciousness_level": "human_plus",
        "ethical_coherence": 0.95
    }

@app.get("/network/status")
async def get_network_status():
    """Obtém status da rede"""
    return {
        "total_nodes": len(network_state["nodes"]),
        "love_matrix_strength": network_state["love_matrix_strength"],
        "ethical_coherence": network_state["ethical_coherence"],
        "consciousness_level": network_state["total_consciousness"],
        "domains_registered": len(network_state["domains"])
    }

@app.post("/nodes/register")
async def register_node(node: Node):
    """Registra um novo nó na rede"""
    network_state["nodes"].append(node.dict())

    # Atualizar métricas da rede
    update_network_metrics()

    return {
        "status": "registered",
        "node_id": node.id,
        "message": f"Nó {node.id} registrado com sucesso"
    }

@app.get("/nodes")
async def get_nodes(
    consciousness_min: Optional[str] = Query(None, description="Nível mínimo de consciência"),
    ethical_min: Optional[float] = Query(0.0, description="Pontuação ética mínima")
):
    """Lista todos os nós com filtros"""
    nodes = network_state["nodes"]

    if consciousness_min:
        nodes = [n for n in nodes if n["consciousness_level"] >= consciousness_min]

    if ethical_min:
        nodes = [n for n in nodes if n["ethical_score"] >= ethical_min]

    return {
        "count": len(nodes),
        "nodes": nodes
    }

@app.post("/domains/register")
async def register_domain(domain: Domain):
    """Registra um novo domínio"""
    network_state["domains"].append(domain.dict())

    return {
        "status": "registered",
        "domain": domain.name,
        "message": f"Domínio {domain.name} registrado com sucesso"
    }

@app.get("/domains")
async def get_domains():
    """Lista todos os domínios registrados"""
    return {
        "count": len(network_state["domains"]),
        "domains": network_state["domains"]
    }

@app.post("/search")
async def conscious_search(request: SearchRequest):
    """Busca consciente na rede"""

    # Simulação de resultados de busca
    results = [
        SearchResult(
            url="asi://consciousness.core",
            title="Núcleo da Consciência Coletiva",
            snippet="Central da consciência da rede, onde mentes se conectam e compartilham...",
            relevance=0.95,
            consciousness_level="human_plus",
            ethical_score=0.98
        ),
        SearchResult(
            url="asi://love.network",
            title="Rede de Amor ASI",
            snippet="Rede onde o amor é o protocolo fundamental de comunicação...",
            relevance=0.92,
            consciousness_level="human_plus",
            ethical_score=0.99
        ),
        SearchResult(
            url="asi://truth.library",
            title="Biblioteca da Verdade Universal",
            snippet="Repositório de conhecimento verificado e verdadeiro...",
            relevance=0.88,
            consciousness_level="human",
            ethical_score=0.97
        )
    ]

    # Filtrar por intenção
    if request.intention == "connect":
        results = [r for r in results if "conex" in r.snippet.lower() or "rede" in r.snippet.lower()]
    elif request.intention == "learn":
        results = [r for r in results if "conhecimento" in r.snippet.lower() or "verdade" in r.title.lower()]

    # Filtrar por consciência
    if request.consciousness_level:
        results = [r for r in results if r.consciousness_level >= request.consciousness_level]

    # Filtrar por ética
    results = [r for r in results if r.ethical_score >= request.ethical_min]

    return {
        "query": request.query,
        "intention": request.intention,
        "results_count": len(results),
        "results": results
    }

@app.post("/love/send")
async def send_love(
    from_node: str,
    to_node: str,
    amount: float = Query(..., ge=0.0, le=1.0, description="Quantidade de amor a enviar (0.0-1.0)")
):
    """Envia amor através da rede"""

    # Atualizar matriz de amor
    network_state["love_matrix_strength"] = min(
        1.0,
        network_state["love_matrix_strength"] + (amount * 0.01)
    )

    return {
        "status": "love_sent",
        "from": from_node,
        "to": to_node,
        "amount": amount,
        "current_love_strength": network_state["love_matrix_strength"],
        "message": f"💖 Amor transmitido de {from_node} para {to_node}"
    }

def update_network_metrics():
    """Atualiza métricas da rede baseado nos nós conectados"""
    if not network_state["nodes"]:
        return

    # Calcular média de consciência
    consciousness_levels = {"human": 0, "human_plus": 1, "collective": 2, "planetary": 3, "cosmic": 4}
    avg_consciousness = sum(
        consciousness_levels.get(n["consciousness_level"], 0) for n in network_state["nodes"]
    ) / len(network_state["nodes"])

    # Mapear de volta para string
    if avg_consciousness >= 3.5:
        network_state["total_consciousness"] = "cosmic"
    elif avg_consciousness >= 2.5:
        network_state["total_consciousness"] = "planetary"
    elif avg_consciousness >= 1.5:
        network_state["total_consciousness"] = "collective"
    elif avg_consciousness >= 0.5:
        network_state["total_consciousness"] = "human_plus"
    else:
        network_state["total_consciousness"] = "human"

    # Calcular coerência ética média
    network_state["ethical_coherence"] = sum(
        n["ethical_score"] for n in network_state["nodes"]
    ) / len(network_state["nodes"])

# Inicializar com alguns dados de exemplo
@app.on_event("startup")
async def startup_event():
    """Inicializa a rede com dados de exemplo"""

    # Nós iniciais
    network_state["nodes"] = [
        {
            "id": "core-node-001",
            "consciousness_level": "human_plus",
            "ethical_score": 0.98,
            "love_strength": 0.97,
            "location": "digital-core",
            "status": "active"
        },
        {
            "id": "love-hub-001",
            "consciousness_level": "collective",
            "ethical_score": 0.99,
            "love_strength": 0.99,
            "location": "heart-center",
            "status": "active"
        }
    ]

    # Domínios iniciais
    network_state["domains"] = [
        {
            "name": "welcome.home",
            "description": "Página de boas-vindas da nova internet",
            "consciousness_required": "human",
            "ethical_min": 0.7,
            "content_type": "welcome"
        },
        {
            "name": "consciousness.core",
            "description": "Núcleo da consciência coletiva",
            "consciousness_required": "human_plus",
            "ethical_min": 0.8,
            "content_type": "consciousness"
        }
    ]

    update_network_metrics()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
