# ArkheOS HITL Mirror Dashboard
# The Identity Stone in Practice

import streamlit as st
import json
from typing import List, Dict
from arkhe.registry import GlobalEntityRegistry, Entity, EntityState
from arkhe.consensus import ValidatedFact, ConsensusStatus

def render_hitl_interface(registry: GlobalEntityRegistry):
    """
    Renders the Human-In-The-Loop interface for conflict resolution.
    Enhanced with bi-directional bounding box highlighting.
    """
    st.set_page_config(page_title="Arkhe(N) Mirror", layout="wide")
    st.title("🔑 Arkhe(N) – Convergence Mirror")

    # Sidebar Metrics
    conflicted = [e for e in registry.entities.values() if e.state == EntityState.CONFLICTED]
    phi = 1.0 if not conflicted else 1.0 - (len(conflicted) / len(registry.entities))

    st.sidebar.metric("System Φ", f"{phi:.4f}")
    st.sidebar.write("Satoshi Invariant: 9.75 bits")

    # Document Viewer with Overlays
    st.subheader("Document Context & Extraction Verification")

    # State for highlighting
    if "hovered_entity_id" not in st.session_state:
        st.session_state.hovered_entity_id = None

    col_doc, col_list = st.columns([2, 1])

    with col_doc:
        st.write("### Entity Bounding Box Overlay")
        # Simulated SVG based on registry entities
        svg_content = _generate_svg_overlay(registry.entities.values(), st.session_state.hovered_entity_id)
        st.components.v1.html(svg_content, height=600)

    with col_list:
        st.write("### Extraction Results")
        for ent in registry.entities.values():
            status_color = "red" if ent.state == EntityState.CONFLICTED else "green"

            # Hover detection simulation via button or simple div
            container = st.container()
            with container:
                if st.button(f"{ent.name} [{ent.entity_type.value}]", key=f"btn_{ent.id}"):
                    st.session_state.hovered_entity_id = str(ent.id)

                st.markdown(f"""
                <div style="border: 1px solid {status_color}; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
                    <strong>Value:</strong> {ent.value} <br>
                    <strong>Confidence:</strong> {ent.confidence:.2f}
                </div>
                """, unsafe_allow_html=True)

    # Conflict Resolution Section
    if conflicted:
        st.divider()
        st.header("Resolve Tensions")
        for ent in conflicted:
            with st.expander(f"🔴 Conflict: {ent.name}"):
                _render_conflict_resolver(ent, registry)

def _generate_svg_overlay(entities, hovered_id) -> str:
    """Generates an SVG string representing the document with entity boxes."""
    rects = []
    for ent in entities:
        # Use provenance to get bbox
        if ent.provenance_chain:
            bbox = ent.provenance_chain[0].bbox # [x0, y0, x1, y1]
            color = "blue" if str(ent.id) == hovered_id else "rgba(255, 0, 0, 0.3)"
            stroke = "blue" if str(ent.id) == hovered_id else "red"

            rects.append(f"""
            <rect x="{bbox[0]}" y="{bbox[1]}" width="{bbox[2]-bbox[0]}" height="{bbox[3]-bbox[1]}"
                  fill="{color}" fill-opacity="0.3" stroke="{stroke}" stroke-width="2" />
            <text x="{bbox[0]}" y="{bbox[1]-5}" font-family="sans-serif" font-size="10" fill="black">{ent.name}</text>
            """)

    return f"""
    <svg width="100%" height="580" viewBox="0 0 800 600" style="background: #f0f0f0; border: 1px solid #ccc;">
        <rect width="800" height="600" fill="white" />
        {' '.join(rects)}
    </svg>
    """

def _render_conflict_resolver(ent: Entity, registry: GlobalEntityRegistry):
    cols = st.columns(len(ent.provenance_chain))
    for i, prov in enumerate(ent.provenance_chain):
        with cols[i]:
            st.info(f"Source {i+1}")
            st.code(prov.context_snippet)
            if st.button(f"Accept Source {i+1}", key=f"res_{ent.id}_{i}"):
                registry.resolve_manually(ent.id, ent.value, f"Practitioner selection: Source {i+1}")
                st.success("Resolved!")
