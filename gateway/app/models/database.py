from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Index, Text
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
import json

Base = declarative_base()

class AgentDB(Base):
    __tablename__ = 'agents'

    id = Column(Integer, primary_key=True)
    agent_id = Column(String(255), unique=True, index=True, nullable=False)
    vk_ref_bio = Column(Float, default=0.5)
    vk_ref_aff = Column(Float, default=0.5)
    vk_ref_soc = Column(Float, default=0.5)
    vk_ref_cog = Column(Float, default=0.5)
    created_at = Column(DateTime, default=datetime.utcnow)

    layers = relationship("StateLayerDB", back_populates="agent")

    def to_dict(self):
        return {
            "agent_id": self.agent_id,
            "vk_ref": {
                "bio": self.vk_ref_bio,
                "aff": self.vk_ref_aff,
                "soc": self.vk_ref_soc,
                "cog": self.vk_ref_cog
            },
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class StateLayerDB(Base):
    __tablename__ = 'state_layers'

    id = Column(Integer, primary_key=True)
    agent_id = Column(String(255), ForeignKey('agents.agent_id'), nullable=False)
    layer_index = Column(Integer, nullable=False)

    bio = Column(Float, nullable=False)
    aff = Column(Float, nullable=False)
    soc = Column(Float, nullable=False)
    cog = Column(Float, nullable=False)

    q_value = Column(Float, nullable=False)
    delta_k = Column(Float, nullable=False)
    t_kr = Column(Float, nullable=False)

    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    agent = relationship("AgentDB", back_populates="layers")

    __table_args__ = (
        Index('idx_agent_layer', 'agent_id', 'layer_index', unique=True),
    )

    @property
    def is_stable(self):
        return self.delta_k < 0.1

    @property
    def is_crisis(self):
        return self.delta_k > 0.5

    @property
    def is_approaching_singularity(self):
        # Miller Limit: φ_q = 4.64. Proxy: Q * 5.0
        return (self.q_value * 5.0) > 4.64

    def to_dict(self):
        return {
            "agent_id": self.agent_id,
            "layer_index": self.layer_index,
            "vk": {"bio": self.bio, "aff": self.aff, "soc": self.soc, "cog": self.cog},
            "q_value": self.q_value,
            "delta_k": self.delta_k,
            "t_kr": self.t_kr,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "is_stable": self.is_stable,
            "is_crisis": self.is_crisis,
            "is_approaching_singularity": self.is_approaching_singularity
        }

class BifurcationDB(Base):
    __tablename__ = 'bifurcations'

    id = Column(Integer, primary_key=True)
    agent_id = Column(String(255), ForeignKey('agents.agent_id'), nullable=False)
    layer_index = Column(Integer, nullable=False)

    delta_k_before = Column(Float, nullable=False)
    delta_k_after = Column(Float, nullable=False)
    bifurcation_type = Column(String(50), nullable=False) # e.g., ENTER_CRISIS, EXIT_CRISIS

    timestamp = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "agent_id": self.agent_id,
            "layer_index": self.layer_index,
            "delta_k_before": self.delta_k_before,
            "delta_k_after": self.delta_k_after,
            "bifurcation_type": self.bifurcation_type,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }

class ConstitutionalAuditDB(Base):
    __tablename__ = 'constitutional_audit'

    id = Column(Integer, primary_key=True)
    method = Column(String(10), nullable=False)
    path = Column(String(255), nullable=False)
    status_code = Column(Integer, nullable=False)

    h_value = Column(Float, nullable=False) # Constante Elena
    processing_time_ms = Column(Float, nullable=False)
    constitutional_status = Column(String(50), nullable=False)

    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            "method": self.method,
            "path": self.path,
            "status_code": self.status_code,
            "h_value": self.h_value,
            "processing_time_ms": self.processing_time_ms,
            "constitutional_status": self.constitutional_status,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }

class SynchronicityLogDB(Base):
    __tablename__ = 'synchronicity_log'

    id = Column(Integer, primary_key=True)
    s_index = Column(Float, nullable=False, index=True)
    status = Column(String(50), nullable=False)

    delta_k_avg = Column(Float, nullable=False)
    p_ac_proxy = Column(Float, nullable=False)
    total_agents = Column(Integer, nullable=False)
    ghost_count = Column(Integer, nullable=False)
    lambda_sync = Column(Float, nullable=False)
    h_value = Column(Float, nullable=False)

    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            "s_index": self.s_index,
            "status": self.status,
            "delta_k_avg": self.delta_k_avg,
            "p_ac_proxy": self.p_ac_proxy,
            "total_agents": self.total_agents,
            "ghost_count": self.ghost_count,
            "lambda_sync": self.lambda_sync,
            "h_value": self.h_value,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
