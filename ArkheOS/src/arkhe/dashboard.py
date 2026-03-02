# ArkheOS HITL Mirror Dashboard
# The Identity Stone in Practice

import streamlit as st
from typing import List
from arkhe.registry import GlobalEntityRegistry, Entity, EntityState
from arkhe.consensus import ValidatedFact, ConsensusStatus

def render_hitl_interface(registry: GlobalEntityRegistry):
    """
    Renders the Human-In-The-Loop interface for conflict resolution.
    This is the technical manifestation of the 'Hesitation' principle.
    """
    st.set_page_config(page_title="Arkhe(N) Mirror", layout="wide")
    st.title("🔑 Arkhe(N) – Convergence Mirror")

    # Calculate System Phi (Φ)
    conflicted = [e for e in registry.entities.values() if e.state == EntityState.CONFLICTED]
    phi = 1.0 if not conflicted else 1.0 - (len(conflicted) / len(registry.entities))

    st.sidebar.metric("System Φ", f"{phi:.4f}")
    st.sidebar.write("Satoshi Invariant: 7.27 bits")

    if not conflicted:
        st.success("Absolute Convergence Achieved (Φ = 1.0000). The arch is stable.")
    else:
        st.warning(f"Detected {len(conflicted)} points of tension in the Geodesic Arch.")

    for ent in conflicted:
        with st.expander(f"🔴 Conflict: {ent.name} ({ent.entity_type.value})"):
            st.write(f"Current Value: {ent.value} {ent.unit or ''}")
            st.subheader("Provenance Evidence Heatmap")

            cols = st.columns(len(ent.provenance_chain))
            for i, prov in enumerate(ent.provenance_chain):
                with cols[i]:
                    st.info(f"Source {i+1} (Page {prov.page})")
                    st.code(prov.context_snippet)
                    if st.button(f"Validate Source {i+1}", key=f"{ent.id}_{i}"):
                        # Apply the Practitioner's Signature (Manual Resolution)
                        registry.resolve_manually(ent.id, ent.value, "Practitioner Validation")
                        st.success(f"Fact resolved as Source {i+1}. Recalculating Φ...")
                        # In a real app, we would use st.rerun()

    st.divider()
    st.subheader("Operational Geometry")
    st.write("Each node in the graph below represents a load-bearing extraction fact.")
    # In a real implementation, this would use st.graphviz_chart or a d3.js component
    st.info("Interactive Geodesic Graph - [PLACEHOLDER]")

if __name__ == "__main__":
    # Sample execution logic for dashboard
    if 'registry' not in st.session_state:
        st.session_state.registry = GlobalEntityRegistry()
    render_hitl_interface(st.session_state.registry)
# arkhe/dashboard.py
from .telemetry import TelemetryCollector

class MetricsDashboard:
    """
    Visualização das métricas C/F em tempo real.
    """

    def render(self, telemetry: TelemetryCollector):
        stats = telemetry.get_stats()

        print("\n" + "╔" + "═" * 60 + "╗")
        print("║" + "      ARKHE(n) TELEMETRY DASHBOARD v2.0".center(60) + "║")
        print("╠" + "═" * 60 + "╣")

        for provider, data in stats.items():
            # Usar disponibilidade ou taxa de sucesso como proxy de C
            C = data.get("availability", 0.0)
            F = 1.0 - C

            bar_len = 20
            filled_c = int(C * bar_len)
            bar_c = "█" * filled_c + "░" * (bar_len - filled_c)

            filled_f = int(F * bar_len)
            bar_f = "█" * filled_f + "░" * (bar_len - filled_f)

            print(f"║ {provider.upper():12} │ C: {bar_c} {C:.2f} │ F: {bar_f} {F:.2f} ║")
            print(f"║             │ Latency: {data.get('avg_latency_ms', 0):.0f}ms │ Calls: {data.get('total_calls', 0):<5} ║")
            print("╠" + "─" * 60 + "╣")

        print("║ Conservation Law: C + F = 1.0 ✅".center(60) + "║")
        print("╚" + "═" * 60 + "╝\n")

if __name__ == "__main__":
    from .telemetry import LLMMetrics, Provider
    import asyncio

    async def demo():
        collector = TelemetryCollector()
        await collector.record(LLMMetrics(Provider.GEMINI, "gen", 120.0, True, 10, 50))
        await collector.record(LLMMetrics(Provider.OLLAMA, "gen", 850.0, True, 10, 40))
        await collector.record(LLMMetrics(Provider.GEMINI, "gen", 150.0, False, 10, 0))

        dashboard = MetricsDashboard()
        dashboard.render(collector)

    asyncio.run(demo())
