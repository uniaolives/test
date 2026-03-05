import React, { useState, useEffect, useRef } from 'react';

interface Message {
  role: 'user' | 'agent' | 'system';
  content: string;
}

interface ChatProps {
  onSendMessage: (message: string) => Promise<string | undefined>;
  onStateChange: (newVk: any) => void;
  vkState: any;
}

export const Chat: React.FC<ChatProps> = ({ onSendMessage, onStateChange, vkState }) => {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'system', content: 'Arkhe(n) Ω+225 Dashboard initialized.' },
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
      background: 'rgba(0, 20, 0, 0.8)',
      border: '1px solid #00ffcc',
      display: 'flex',
      flexDirection: 'column',
      color: '#00ffcc',
      boxShadow: '0 0 20px rgba(0, 255, 204, 0.2)',
      zIndex: 100,
    }}>
      <div style={{ padding: '10px', borderBottom: '1px solid #00ffcc', fontSize: '12px' }}>
        CORUS AGENT - L6 METACOGNITION
        <br />
        Q: {vkState.q_permeability.toFixed(3)} | ΔK: {(1 - vkState.q_permeability).toFixed(3)}
      </div>

      <div ref={scrollRef} style={{ flex: 1, overflowY: 'auto', padding: '10px', fontSize: '14px' }}>
        {messages.map((m, i) => (
          <div key={i} style={{ marginBottom: '10px', opacity: m.role === 'system' ? 0.7 : 1 }}>
            <strong>{m.role === 'user' ? 'YOU > ' : m.role === 'agent' ? 'CORUS > ' : 'SYS > '}</strong>
            {m.content}
          </div>
        ))}
      </div>

      <div style={{ padding: '10px', borderTop: '1px solid #00ffcc', display: 'flex' }}>
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
        <button onClick={handleSend} style={{ background: 'transparent', border: '1px solid #00ffcc', color: '#00ffcc', cursor: 'pointer' }}>SEND</button>
      </div>

      <div style={{ padding: '10px', display: 'flex', gap: '5px' }}>
         {['bio', 'aff', 'soc', 'cog'].map(key => (
           <div key={key} style={{ flex: 1 }}>
             <label style={{ fontSize: '10px' }}>{key.toUpperCase()}</label>
             <input
               type="range" min="0" max="1" step="0.01"
               value={vkState[key]}
               onChange={(e) => onStateChange({ [key]: parseFloat(e.target.value) })}
               style={{ width: '100%' }}
             />
           </div>
         ))}
      </div>
    </div>
  );
};
