from sqlalchemy.orm import Session
from sqlalchemy import func
from ..models.database import AgentDB, StateLayerDB, BifurcationDB, ConstitutionalAuditDB, SynchronicityLogDB
from datetime import datetime
from typing import List, Optional, Dict, Any

class DMRRepository:
    def __init__(self, db: Session):
        self.db = db

    # --- Agent Operations ---

    def create_agent(self, agent_id: str, vk_ref: Dict[str, float]) -> AgentDB:
        agent = AgentDB(
            agent_id=agent_id,
            vk_ref_bio=vk_ref.get('bio', 0.5),
            vk_ref_aff=vk_ref.get('aff', 0.5),
            vk_ref_soc=vk_ref.get('soc', 0.5),
            vk_ref_cog=vk_ref.get('cog', 0.5)
        )
        self.db.add(agent)
        self.db.commit()
        self.db.refresh(agent)
        return agent

    def get_agent(self, agent_id: str) -> Optional[AgentDB]:
        return self.db.query(AgentDB).filter(AgentDB.agent_id == agent_id).first()

    def agent_exists(self, agent_id: str) -> bool:
        return self.db.query(AgentDB).filter(AgentDB.agent_id == agent_id).count() > 0

    def list_agents(self, skip: int = 0, limit: int = 100) -> List[AgentDB]:
        return self.db.query(AgentDB).offset(skip).limit(limit).all()

    def count_agents(self) -> int:
        return self.db.query(AgentDB).count()

    # --- State Layer Operations ---

    def add_layer(self, agent_id: str, bio: float, aff: float, soc: float, cog: float, q_value: float, delta_k: float, t_kr: float) -> StateLayerDB:
        # Get latest layer index
        latest_layer = self.get_latest_layer(agent_id)
        next_index = (latest_layer.layer_index + 1) if latest_layer else 0

        # Check for bifurcation before adding
        if latest_layer:
            self._record_bifurcation(agent_id, next_index, latest_layer.delta_k, delta_k)

        layer = StateLayerDB(
            agent_id=agent_id,
            layer_index=next_index,
            bio=bio, aff=aff, soc=soc, cog=cog,
            q_value=q_value,
            delta_k=delta_k,
            t_kr=t_kr
        )
        self.db.add(layer)
        self.db.commit()
        self.db.refresh(layer)
        return layer

    def get_layer(self, agent_id: str, layer_index: int) -> Optional[StateLayerDB]:
        return self.db.query(StateLayerDB).filter(
            StateLayerDB.agent_id == agent_id,
            StateLayerDB.layer_index == layer_index
        ).first()

    def get_latest_layer(self, agent_id: str) -> Optional[StateLayerDB]:
        return self.db.query(StateLayerDB).filter(
            StateLayerDB.agent_id == agent_id
        ).order_by(StateLayerDB.layer_index.desc()).first()

    def get_trajectory(self, agent_id: str, limit: int = 1000) -> List[StateLayerDB]:
        return self.db.query(StateLayerDB).filter(
            StateLayerDB.agent_id == agent_id
        ).order_by(StateLayerDB.layer_index.asc()).limit(limit).all()

    def count_layers(self, agent_id: str) -> int:
        return self.db.query(StateLayerDB).filter(StateLayerDB.agent_id == agent_id).count()

    # --- Bifurcation Operations ---

    def _record_bifurcation(self, agent_id: str, layer_index: int, delta_k_before: float, delta_k_after: float):
        b_type = None
        if delta_k_before <= 0.5 and delta_k_after > 0.5:
            b_type = "ENTER_CRISIS"
        elif delta_k_before > 0.5 and delta_k_after <= 0.5:
            b_type = "EXIT_CRISIS"
        elif abs(delta_k_after - delta_k_before) > 0.3:
            b_type = "SUDDEN_SHIFT"

        if b_type:
            bifurcation = BifurcationDB(
                agent_id=agent_id,
                layer_index=layer_index,
                delta_k_before=delta_k_before,
                delta_k_after=delta_k_after,
                bifurcation_type=b_type
            )
            self.db.add(bifurcation)
            # Commit is handled by the caller (add_layer)

    def get_bifurcations(self, agent_id: Optional[str] = None, limit: int = 100) -> List[BifurcationDB]:
        query = self.db.query(BifurcationDB)
        if agent_id:
            query = query.filter(BifurcationDB.agent_id == agent_id)
        return query.order_by(BifurcationDB.timestamp.desc()).limit(limit).all()

    def count_bifurcations(self, agent_id: Optional[str] = None) -> int:
        query = self.db.query(BifurcationDB)
        if agent_id:
            query = query.filter(BifurcationDB.agent_id == agent_id)
        return query.count()

    # --- System Metrics ---

    def compute_system_metrics(self) -> Dict[str, Any]:
        # Average delta_k across latest layers of all agents
        subquery = self.db.query(
            StateLayerDB.agent_id,
            func.max(StateLayerDB.layer_index).label('max_idx')
        ).group_by(StateLayerDB.agent_id).subquery()

        latest_layers = self.db.query(StateLayerDB).join(
            subquery,
            (StateLayerDB.agent_id == subquery.c.agent_id) &
            (StateLayerDB.layer_index == subquery.c.max_idx)
        ).all()

        if not latest_layers:
            return {"avg_delta_k": 0.0, "avg_q": 0.0, "total_agents": 0}

        avg_delta_k = sum(l.delta_k for l in latest_layers) / len(latest_layers)
        avg_q = sum(l.q_value for l in latest_layers) / len(latest_layers)

        return {
            "avg_delta_k": avg_delta_k,
            "avg_q": avg_q,
            "total_agents": len(latest_layers)
        }

    def compute_h_value(self) -> float:
        # Constante Elena (H ≤ 1). Proxy from last 100 audit entries
        avg_h = self.db.query(func.avg(ConstitutionalAuditDB.h_value)).scalar()
        return float(avg_h) if avg_h is not None else 0.5

    # --- Constitutional Audit ---

    def log_constitutional_action(self, method: str, path: str, status_code: int, h_value: float, processing_time_ms: float, status: str) -> ConstitutionalAuditDB:
        audit = ConstitutionalAuditDB(
            method=method, path=path, status_code=status_code,
            h_value=h_value, processing_time_ms=processing_time_ms,
            constitutional_status=status
        )
        self.db.add(audit)
        self.db.commit()
        return audit

    def get_audit_trail(self, limit: int = 100) -> List[ConstitutionalAuditDB]:
        return self.db.query(ConstitutionalAuditDB).order_by(ConstitutionalAuditDB.timestamp.desc()).limit(limit).all()

    # --- Synchronicity Logging ---

    def log_synchronicity(self, s_index: float, status: str, delta_k_avg: float, p_ac_proxy: float, total_agents: int, ghost_count: int, lambda_sync: float, h_value: float) -> SynchronicityLogDB:
        log = SynchronicityLogDB(
            s_index=s_index, status=status, delta_k_avg=delta_k_avg,
            p_ac_proxy=p_ac_proxy, total_agents=total_agents,
            ghost_count=ghost_count, lambda_sync=lambda_sync, h_value=h_value
        )
        self.db.add(log)
        self.db.commit()
        return log

    def get_synchronicity_history(self, limit: int = 100) -> List[SynchronicityLogDB]:
        return self.db.query(SynchronicityLogDB).order_by(SynchronicityLogDB.timestamp.desc()).limit(limit).all()

    def get_peak_synchronicity(self) -> Optional[SynchronicityLogDB]:
        return self.db.query(SynchronicityLogDB).order_by(SynchronicityLogDB.s_index.desc()).first()
