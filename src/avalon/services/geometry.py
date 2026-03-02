# src/avalon/services/geometry.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.python.axos.axos_v3 import AxosV3
import uvicorn

app = FastAPI(title="Arkhe Geometry of Consciousness Service")
axos = AxosV3()

class ExploreRequest(BaseModel):
    h11: int
    h21: int
    steps: int = 1

@app.post("/quantum/explore/moduli_space")
async def explore_moduli(req: ExploreRequest):
    try:
        result = axos.explore_landscape(h11=req.h11, h21=req.h21)
        return {"status": "success", "data": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quantum/generate/entity")
async def generate_entity():
    try:
        result = axos.generate_entity()
        return {"status": "success", "data": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
