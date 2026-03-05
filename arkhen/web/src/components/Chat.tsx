import React, { useState, useEffect, useRef } from 'react';

interface Message {
  role: 'user' | 'agent' | 'system';
  content: string;
}

interface ChatProps {
  onSendMessage: (message: string) => Promise<string | undefined>;
  onStateChange: (newVk: any) => void;
  vkState: any;
  nodeState: any;
}

export const Chat: React.FC<ChatProps> = ({ onSendMessage, onStateChange, vkState, nodeState }) => {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'system', content: 'Arkhe(n) Mycelium Dashboard initialized.' },
  ]);
  const [input, setInput] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMsg: Message = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');

    const agentResponse = await onSendMessage(input);
    if (agentResponse) {
      setMessages((prev) => [...prev, { role: 'agent', content: agentResponse }]);
    }
  };

  return (
    <div style={{
      position: 'absolute',
      bottom: '20px',
      right: '20px',
      width: '400px',
      height: '500px',
      background: 'rgba(0, 20, 10, 0.85)',
      border: '1px solid #00ffcc',
      display: 'flex',
      flexDirection: 'column',
      color: '#00ffcc',
      boxShadow: '0 0 20px rgba(0, 255, 204, 0.2)',
      zIndex: 100,
      borderRadius: '8px',
      overflow: 'hidden'
    }}>
      <div style={{ padding: '10px', borderBottom: '1px solid #00ffcc', fontSize: '12px', background: 'rgba(0, 40, 20, 0.5)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
          <span>CORUS AGENT - L6 METACOGNITION</span>
          <span style={{ color: nodeState.isCrisis ? '#ff3366' : '#00ffcc' }}>
            {nodeState.isCrisis ? 'CRISIS MODE' : 'STABLE'}
          </span>
        </div>
        <div style={{ display: 'flex', gap: '10px', fontSize: '10px' }}>
          <span>Q: {nodeState.Q.toFixed(3)}</span>
          <span>ΔK: {nodeState.deltaK.toFixed(3)}</span>
          <span>t_KR: {Math.floor(nodeState.t_KR)}s</span>
        </div>
      </div>

      <div ref={scrollRef} style={{ flex: 1, overflowY: 'auto', padding: '10px', fontSize: '14px', scrollBehavior: 'smooth' }}>
        {messages.map((m, i) => (
          <div key={i} style={{ marginBottom: '10px', opacity: m.role === 'system' ? 0.7 : 1 }}>
            <strong style={{ color: m.role === 'user' ? '#00ffcc' : m.role === 'agent' ? '#aaffaa' : '#888' }}>
              {m.role === 'user' ? 'YOU > ' : m.role === 'agent' ? 'CORUS > ' : 'SYS > '}
            </strong>
            {m.content}
          </div>
        ))}
      </div>

      <div style={{ padding: '10px', borderTop: '1px solid #00ffcc', display: 'flex', background: 'rgba(0,0,0,0.3)' }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Transmit intention..."
          style={{
            flex: 1,
            background: 'transparent',
            border: 'none',
            color: '#00ffcc',
            outline: 'none',
            fontFamily: 'inherit',
          }}
        />
        <button
          onClick={handleSend}
          style={{
            background: 'transparent', border: '1px solid #00ffcc', color: '#00ffcc',
            cursor: 'pointer', padding: '5px 15px', fontWeight: 'bold'
          }}
        >
          SEND
        </button>
      </div>

      <div style={{ padding: '10px', display: 'flex', gap: '8px', background: 'rgba(0, 30, 15, 0.4)' }}>
         {['bio', 'aff', 'soc', 'cog'].map(key => (
           <div key={key} style={{ flex: 1 }}>
             <label style={{ fontSize: '9px', display: 'block', marginBottom: '2px', color: '#888' }}>{key.toUpperCase()}</label>
             <input
               type="range" min="0" max="1" step="0.01"
               value={vkState[key]}
               onChange={(e) => onStateChange({ [key]: parseFloat(e.target.value) })}
               style={{ width: '100%', accentColor: '#00ffcc' }}
             />
           </div>
         ))}
      </div>
    </div>
  );
};
