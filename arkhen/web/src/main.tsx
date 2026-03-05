import React, { useEffect, useRef, useState } from 'react';
import ReactDOM from 'react-dom/client';
import { TorusVisualizer, KatharosVector } from './visualizer';
import { Chat } from './components/Chat';
import { ArkhenAgent } from './agent';

// @ts-ignore
import neuroceptionShader from './shaders/neuroception.wgsl?raw';

const App: React.FC = () => {
  const visualizerContainerRef = useRef<HTMLDivElement>(null);
  const visualizerRef = useRef<TorusVisualizer | null>(null);
  const [vkState, setVkState] = useState<KatharosVector>({
    bio: 0.618,
    aff: 0.5,
    soc: 0.3,
    cog: 0.4,
    q_permeability: 0.95,
  });

  const [agentStatus, setAgentStatus] = useState<string>('Initializing...');
  const agentRef = useRef<ArkhenAgent | null>(null);
  const [remoteStatus, setRemoteStatus] = useState<string>('DISCONNECTED');

  useEffect(() => {
    // WebSocket Connection for Remote Orchestration
    const socket = new WebSocket('ws://localhost:8000/ws/vk');

    socket.onopen = () => {
      console.log("[Ω] Connected to Arkhe(n) Gateway");
      setRemoteStatus('CONNECTED');
      socket.send(JSON.stringify({ type: 'HEARTBEAT', timestamp: Date.now() }));
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'MATTER_BEAM_SYNC') {
           console.log("[Ω] Remote Sync Received:", data);
        }
      } catch (e) {
        console.log("[Ω] Received from Gateway:", event.data);
      }
    };

    socket.onerror = (error) => {
      console.warn("[Ω] Gateway unreachable. Running in Isolated Mode.");
      setRemoteStatus('ISOLATED');
    };

    socket.onclose = () => {
      setRemoteStatus('DISCONNECTED');
    };

    return () => socket.close();
  }, []);

  useEffect(() => {
    const init = async () => {
      if (visualizerContainerRef.current && !visualizerRef.current) {
        visualizerRef.current = new TorusVisualizer(visualizerContainerRef.current);
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

  useEffect(() => {
    if (visualizerRef.current) {
      visualizerRef.current.updateVK(vkState).then(() => {
          const actualVk = visualizerRef.current?.getVK();
          if (actualVk && actualVk.q_permeability !== vkState.q_permeability) {
              setVkState(prev => ({ ...prev, q_permeability: actualVk.q_permeability }));
          }
      });
    }
  }, [vkState.bio, vkState.aff, vkState.soc, vkState.cog]);

  const handleSendMessage = async (msg: string) => {
    if (!agentRef.current) return;
    return await agentRef.current.chat(msg, vkState);
  };

  const updateState = (newVk: Partial<KatharosVector>) => {
    setVkState((prev) => ({ ...prev, ...newVk }));
  };

  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh', overflow: 'hidden' }}>
      <div
        ref={visualizerContainerRef}
        style={{ width: '100%', height: '100%', cursor: 'crosshair' }}
      />

      <div style={{ position: 'absolute', top: '20px', left: '20px', color: '#00ffcc', pointerEvents: 'none' }}>
        <h1 style={{ margin: 0, textShadow: '0 0 10px #00ffcc' }}>⚛️ ARKHE(N) DASHBOARD</h1>
        <div style={{ fontSize: '12px', background: 'rgba(0,0,0,0.5)', padding: '10px', border: '1px solid #00ffcc', marginTop: '10px' }}>
          NODE IDENT: UNGOVERNABLE_VRAM_0x7F3B...<br />
          SUBSTRATE: WebGPU/WebLLM Layer -1 to 6<br />
          λ_sync: {(vkState.q_permeability * 0.99).toFixed(4)} Hz<br />
          GATEWAY: {remoteStatus}<br />
          AGENT STATUS: {agentStatus}
        </div>
      </div>

      <Chat
        onSendMessage={handleSendMessage}
        onStateChange={updateState}
        vkState={vkState}
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
