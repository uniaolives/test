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

            # Entity Heatmap Integration
            st.subheader("Layout Heatmap Overlay")

            # Simple SVG Heatmap
            svg_overlay = """
            <svg width="600" height="400" style="background: #eee; border: 1px solid #ccc;">
                <rect x="120" y="50" width="200" height="30" fill="red" fill-opacity="0.3" stroke="red" />
                <text x="125" y="70" font-family="sans-serif" font-size="12" fill="black">Extracted Value Area</text>
            </svg>
            """
            st.components.v1.html(svg_overlay, height=420)

            cols = st.columns(len(ent.provenance_chain))
            for i, prov in enumerate(ent.provenance_chain):
                with cols[i]:
                    st.info(f"Source {i+1} (Page {prov.page})")
                    st.code(prov.context_snippet)

                    # Simulated Heatmap Highlight (SVG representation)
                    st.markdown(f"""
                    <div style="border: 2px solid #00f0ff; padding: 10px; border-radius: 5px; background: rgba(0, 240, 255, 0.1);">
                      <strong>Heatmap Bbox:</strong> {prov.bbox} <br>
                      <span style="color: #70ff70;">[HIGHLIGHTED ON DOC]</span>
                    </div>
                    """, unsafe_allow_html=True)

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
