from pydantic import BaseModel, Field
from typing import List, Optional

class KatharosVector(BaseModel):
    bio: float = Field(..., ge=0, le=1)
    aff: float = Field(..., ge=0, le=1)
    soc: float = Field(..., ge=0, le=1)
    cog: float = Field(..., ge=0, le=1)

class StateLayer(BaseModel):
    timestamp: int
    vk: KatharosVector
    delta_k: float
    q: float
    intensity: float
