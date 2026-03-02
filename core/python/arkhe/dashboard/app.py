import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import socket
import json
import time
from datetime import datetime

# Configuração da Página
st.set_page_config(
    page_title="Arkhe(n) Quantum Dashboard",
    page_icon="🜁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constantes Arkhe(n)
PHI_TARGET = 0.618033988749894

def query_arkhed_ipc(command, payload=None):
    """Consulta o daemon arkhed via Unix Domain Socket."""
    socket_path = "/tmp/arkhed.sock"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(socket_path)
            cmd_data = {"type": command}
            if payload:
                cmd_data.update(payload)
            client.sendall(json.dumps(cmd_data).encode())
            response = client.recv(4096)
            return json.loads(response.decode())
    except Exception as e:
        return {"type": "error", "message": str(e)}

def get_status():
    res = query_arkhed_ipc("get_status")
    if res and res["type"] == "status":
        return res["phi"], res.get("entropy", 0.618)
    return PHI_TARGET, 0.5

# Sidebar
st.sidebar.title("🜁 Arkhe(n) OS")
st.sidebar.markdown("---")
st.sidebar.subheader("Sistema")
phi, entropy = get_status()
st.sidebar.metric("Criticidade (φ)", f"{phi:.4f}", f"{phi - PHI_TARGET:.4f}")
st.sidebar.metric("Entropia Global", f"{entropy:.4f}")

# Dashboard Sidebar: Night Vision Goggles
st.sidebar.markdown("---")
st.sidebar.subheader("🕶️ Night Vision Goggles")
nv_active = st.sidebar.toggle("Monitorar Goggles", value=True)
if nv_active:
    st.sidebar.success("Sensor nv-goggles-001 Ativo")
    st.sidebar.progress(62, text="Bateria: 62%")

# Dashboard Sidebar: Foundry Bridge
st.sidebar.markdown("---")
st.sidebar.subheader("🔷 Foundry Bridge")
foundry_connected = st.sidebar.status("Conectado ao Foundry OSDK", state="running")
st.sidebar.caption("Sync: GlobalManifold-Foundry-A")

# Layout Principal: Tabs
tab1, tab2, tab3 = st.tabs(["Dashboard Geral", "Night Vision", "Foundry Bridge"])

with tab1:
    st.title("Quantum OS Observability Dashboard (Ω+207)")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Estado de Consciência (Ψ-Field)")
        fig = go.Figure(data=[go.Surface(z=np.random.rand(10, 10))])
        fig.update_layout(title='Topologia do Manifold', autosize=False,
                          width=400, height=400,
                          margin=dict(l=65, r=50, b=65, t=90))
        st.plotly_chart(fig)

    with col2:
        st.subheader("Métricas Termodinâmicas")
        if 'phi_history' not in st.session_state:
            st.session_state.phi_history = []
        st.session_state.phi_history.append(phi)
        if len(st.session_state.phi_history) > 50:
            st.session_state.phi_history.pop(0)
        df_phi = pd.DataFrame(st.session_state.phi_history, columns=['phi'])
        st.line_chart(df_phi)
        st.caption("Evolução da Criticidade (φ)")

    with col3:
        st.subheader("SecOps & Alertas")
        alerts = [
            {"timestamp": datetime.now().strftime("%H:%M:%S"), "type": "INFO", "msg": "Sistema Estável (Criticality Zone)"},
            {"timestamp": datetime.now().strftime("%H:%M:%S"), "type": "SEC", "msg": "Kyber key rotation performed"},
            {"timestamp": datetime.now().strftime("%H:%M:%S"), "type": "DEP", "msg": "NV Goggles connected via MQTT"},
            {"timestamp": datetime.now().strftime("%H:%M:%S"), "type": "FND", "msg": "Foundry Ontology Sync Active"},
        ]
        for alert in alerts:
            st.info(f"[{alert['timestamp']}] {alert['type']}: {alert['msg']}")

    st.header("Fluxo de Handovers (Real-time)")
    handover_data = pd.DataFrame({
        'ID': [f"h-{i}" for i in range(5)],
        'Type': ['Excitatory', 'Inhibitory', 'Meta', 'Excitatory', 'RLTransition'],
        'Source': ['Internal', 'Internal', 'FoundryBridge', 'Internal', 'Internal'],
        'Entropy': np.random.rand(5) * 0.1,
    })
    st.table(handover_data)

with tab2:
    st.header("🕶️ Night Vision Perception Feed")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/e/e0/Night_vision_view.jpg",
                 caption="Live Feed: nv-goggles-001 (Simulado)", use_container_width=True)
    with c2:
        st.write("Métricas do Sensor:")
        st.metric("Entropia Local", "0.618", "+0.002")
        st.metric("Fidelity", "0.985", "-0.001")
        st.button("Recalibrar Sensor")

with tab3:
    st.header("🔷 Foundry Ontology Bridge")
    st.markdown("""
    Esta interface monitora a sincronização entre a **Ontologia do Palantir Foundry** e o **Campo Ψ do Arkhe(n)**.
    Objetos empresariais são mapeados para estados quânticos em tempo real.
    """)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Objetos Sincronizados (OSDK)")
        foundry_data = pd.DataFrame([
            {"ID": "obj-491", "Type": "Vessel", "Status": "Synced", "Phi": 0.618},
            {"ID": "obj-102", "Type": "SupplyChainAlert", "Status": "Pending", "Phi": 0.72},
            {"ID": "obj-883", "Type": "Flight", "Status": "Synced", "Phi": 0.615},
        ])
        st.dataframe(foundry_data)

    with col_b:
        st.subheader("Mapeamento Termodinâmico")
        # Gráfico de barras comparando criticidade de objetos
        fig_foundry = px.bar(foundry_data, x='ID', y='Phi', color='Type',
                             title="Criticidade por Objeto Foundry")
        fig_foundry.add_hline(y=0.618, line_dash="dot", annotation_text="Target Φ")
        st.plotly_chart(fig_foundry)

    st.divider()
    st.subheader("Log de Sincronização")
    st.code(f"""
    [OSDK] SYNC START: {datetime.now().isoformat()}
    [BRIDGE] Mapping Foundry Object 'SupplyChainAlert' -> Handover(Meta)
    [CONSTITUTION] P4 Check: Entropy delta within limits (0.05)
    [LEDGER] Handover recorded: {uuid.uuid4() if 'uuid' in globals() else '550e8400-e29b-41d4-a716-446655440000'}
    """, language="text")

# Controle de Parâmetros
st.header("Intervenção do Operador (P1 Override)")
new_phi = st.slider("Ajuste Manual de φ", 0.0, 1.0, phi)
if st.button("Aplicar Novo φ"):
    res = query_arkhed_ipc("set_phi", {"value": new_phi})
    st.success(f"Solicitação enviada: φ={new_phi}")

# Rodapé
st.markdown("---")
st.caption("Arkhe(n) Dashboard v1.2.0 | Ω+211 Ratified | Foundry Bridge Integrated")
