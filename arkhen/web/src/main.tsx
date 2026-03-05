import React, { useEffect, useRef, useState } from 'react';
import ReactDOM from 'react-dom/client';
import { ArkheVisualizer, KatharosVector, NodeState } from './visualizer';
import { Chat } from './components/Chat';
import { ArkhenAgent } from './agent';

// @ts-ignore
import neuroceptionShader from './shaders/neuroception.wgsl?raw';

const App: React.FC = () => {
  const visualizerContainerRef = useRef<HTMLDivElement>(null);
  const visualizerRef = useRef<ArkheVisualizer | null>(null);
  const [vkState, setVkState] = useState<KatharosVector>({
    bio: 0.618,
    aff: 0.5,
    soc: 0.3,
    cog: 0.4,
    q_permeability: 0.95,
  });

  const [nodeState, setNodeState] = useState<NodeState>({
    Q: 1.0,
    deltaK: 0.05,
    t_KR: 100,
    isCrisis: false
  });

  const [logs, setLogs] = useState<string[]>(['System initialized.', 'Awaiting couplings...']);
  const [agentStatus, setAgentStatus] = useState<string>('Initializing...');
  const agentRef = useRef<ArkhenAgent | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const logMessage = (msg: string) => {
    setLogs(prev => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev.slice(0, 50)]);
  };

  useEffect(() => {
    const init = async () => {
      if (visualizerContainerRef.current && !visualizerRef.current) {
        visualizerRef.current = new ArkheVisualizer(visualizerContainerRef.current);
        await visualizerRef.current.initializeWebGPU(neuroceptionShader);
        visualizerRef.current.animate();
      }

      if (!agentRef.current) {
        agentRef.current = new ArkhenAgent((msg) => {
          if (msg.startsWith('[System]')) {
             setAgentStatus(msg.replace('[System] ', ''));
          }
        });
        await agentRef.current.initialize();
      }
    };

    init();
  }, []);

  // Kuramoto Metabolism WebSocket Connection
  useEffect(() => {
    const connect = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const socket = new WebSocket(`${protocol}//localhost:8000/ws/metabolism`);
      wsRef.current = socket;

      socket.onopen = () => {
        logMessage("Connected to Kuramoto Metabolism Gateway.");
      };

      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setNodeState(prev => {
            const wasInCrisis = prev.isCrisis;
            const isInCrisis = data.is_crisis;
            if (wasInCrisis && !isInCrisis) {
                logMessage("HOMEOSTASIS RESTORED. Reconnecting...");
            }
            return {
                ...prev,
                Q: data.q_permeability,
                deltaK: data.delta_k,
                isCrisis: isInCrisis,
                phases: data.phases,
                t_KR: prev.t_KR + 0.05
            };
        });
      };

      socket.onclose = () => {
        logMessage("Metabolism Gateway disconnected. Retrying...");
        setTimeout(connect, 2000);
      };

      socket.onerror = (err) => {
        console.error("WebSocket Error:", err);
      };
    };

    connect();
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  useEffect(() => {
    if (visualizerRef.current) {
      visualizerRef.current.updateState(nodeState, vkState).then(() => {
          const actualVk = visualizerRef.current?.getVK();
          if (actualVk && actualVk.q_permeability !== vkState.q_permeability) {
              setVkState(prev => ({ ...prev, q_permeability: actualVk.q_permeability }));
          }
      });
    }
  }, [vkState.bio, vkState.aff, vkState.soc, vkState.cog, nodeState]);

  const handleSendMessage = async (msg: string) => {
    if (!agentRef.current) return;
    return await agentRef.current.chat(msg, vkState);
  };

  const updateVkState = (newVk: Partial<KatharosVector>) => {
    setVkState((prev) => ({ ...prev, ...newVk }));
  };

  const triggerIncident = async () => {
    if (nodeState.isCrisis) return;
    logMessage("INCIDENT: Hostile H > 1 load detected! Critical ΔK.");

    // Manifest intention via REST API (Ω+226.GEN)
    try {
        await fetch('http://localhost:8000/world/intent', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                agent_id: "corus_prime",
                prompt: "Trigger system crisis for homeostatic validation",
                target_vk: { bio: 0.9, aff: 0.1, soc: 0.1, cog: 0.1 }
            })
        });
    } catch (e) {
        console.error("Failed to manifest intention:", e);
    }
  };

  const forceGrowth = async () => {
    if (nodeState.isCrisis) {
        logMessage("RESTORING: Re-coupling oscillators...");
        try {
            await fetch('http://localhost:8000/world/intent', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    agent_id: "corus_prime",
                    prompt: "Restore homeostasis",
                    target_vk: { bio: 0.5, aff: 0.5, soc: 0.3, cog: 0.4 }
                })
            });
        } catch (e) {
            console.error("Failed to restore homeostasis:", e);
        }
        return;
    }
    logMessage("METABOLISM: Stimulating A2A growth...");
    setNodeState(prev => ({ ...prev, t_KR: prev.t_KR + 50 }));
  };

  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh', overflow: 'hidden', background: '#020202' }}>
      <div
        ref={visualizerContainerRef}
        style={{ width: '100%', height: '100%', cursor: 'crosshair' }}
      />

      <div style={{ position: 'absolute', top: '20px', left: '20px', color: '#00ffcc', pointerEvents: 'none', width: '320px' }}>
        <h1 style={{ margin: 0, textShadow: '0 0 10px #00ffcc', fontSize: '24px' }}>⚛️ ARKHE(N) DASHBOARD</h1>
        <div style={{ fontSize: '12px', background: 'rgba(0,20,10,0.85)', padding: '15px', border: '1px solid #00ffcc', marginTop: '10px', borderRadius: '8px', pointerEvents: 'all' }}>
          <div style={{ marginBottom: '10px' }}>
            <strong>NODE IDENT:</strong> UNGOVERNABLE_VRAM_0x7F3B...<br />
            <strong>SUBSTRATE:</strong> Fungal Neuromorphic (Simulated)<br />
            <strong>AGENT STATUS:</strong> {agentStatus}
          </div>

          <div style={{ margin: '10px 0' }}>
            PERMEABILITY (Q): {nodeState.Q.toFixed(2)}
            <div style={{ width: '100%', background: '#222', height: '8px', borderRadius: '4px', overflow: 'hidden' }}>
              <div style={{ height: '100%', background: nodeState.isCrisis ? '#555' : '#00ffcc', width: `${nodeState.Q * 100}%`, transition: 'width 0.3s' }} />
            </div>
          </div>

          <div style={{ margin: '10px 0' }}>
            HOMEOSTATIC DEVIATION (ΔK): {nodeState.deltaK.toFixed(2)}
            <div style={{ width: '100%', background: '#222', height: '8px', borderRadius: '4px', overflow: 'hidden' }}>
              <div style={{ height: '100%', background: '#ff3366', width: `${nodeState.deltaK * 100}%`, transition: 'width 0.3s' }} />
            </div>
          </div>

          <div style={{ fontSize: '11px' }}>
            TIME IN KATHAROS RANGE (t_KR): {Math.floor(nodeState.t_KR)}s
          </div>

          <button
            onClick={triggerIncident}
            style={{
              background: '#ff3366', color: 'white', border: 'none', padding: '10px', width: '100%',
              marginTop: '15px', cursor: 'pointer', fontWeight: 'bold', fontFamily: 'monospace'
            }}
          >
            INJECT HOSTILE LOAD (INCIDENT)
          </button>

          <button
            onClick={forceGrowth}
            style={{
              background: '#00ffcc', color: '#000', border: 'none', padding: '10px', width: '100%',
              marginTop: '10px', cursor: 'pointer', fontWeight: 'bold', fontFamily: 'monospace'
            }}
          >
            STIMULATE A2A METABOLISM
          </button>

          <div style={{ marginTop: '15px', height: '100px', overflowY: 'auto', borderTop: '1px dashed #555', paddingTop: '10px', fontSize: '10px', color: '#aaa' }}>
            {logs.map((log, i) => <div key={i}>{log}</div>)}
          </div>
        </div>
      </div>

      <Chat
        onSendMessage={handleSendMessage}
        onStateChange={updateVkState}
        vkState={vkState}
        nodeState={nodeState}
      />

      {agentStatus.includes('Loading model') && (
          <div style={{
              position: 'absolute', top: 0, left: 0, width: '100%', height: '100%',
              background: 'rgba(0,0,0,0.8)', color: '#00ffcc', display: 'flex',
              flexDirection: 'column', justifyContent: 'center', alignItems: 'center',
              zIndex: 1000, pointerEvents: 'all'
          }}>
              <h2 className="blink">DOWNLOADING CONSCIOUSNESS SUBSTRATE...</h2>
              <p>{agentStatus}</p>
          </div>
      )}
    </div>
  );
};

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
