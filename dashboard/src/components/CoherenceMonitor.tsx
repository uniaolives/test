// src/components/CoherenceMonitor.tsx
// O "Multiviewer" da Teknet

import React, { useState, useEffect } from 'react';

// Mocking OrbClient and @arkhe/http4-client as it's a new architectural addition
// In a real environment, this would be imported from the library
/* import { OrbClient } from '@arkhe/http4-client'; */

interface StatusLightProps {
  status: 'LOCKED' | 'DRIFT';
}

const StatusLight: React.FC<StatusLightProps> = ({ status }) => (
  <div className={`status-light ${status.toLowerCase()}`}>
    {status}
  </div>
);

export const CoherenceMonitor: React.FC = () => {
  const [lambda, setLambda] = useState(0);

  useEffect(() => {
    // In a real scenario, this would connect to the actual Timechain stream
    const ws = new WebSocket('wss://timechain.arkhe/phase-stream');

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      setLambda(data.lambda_2);

      // Visual Warning for Low Coherence
      if (data.lambda_2 < 0.618) {
        document.body.style.backgroundColor = 'red'; // Decoherence Alert
      } else {
        document.body.style.backgroundColor = ''; // Reset
      }
    };

    // Mocking websocket messages for demonstration if not connected
    const interval = setInterval(() => {
        if (ws.readyState !== WebSocket.OPEN) {
            const mockLambda = 0.9 + Math.random() * 0.1;
            setLambda(mockLambda);
        }
    }, 2000);

    return () => {
        ws.close();
        clearInterval(interval);
    };
  }, []);

  return (
    <div className="monitor">
      <h1>Global Coherence (λ₂)</h1>
      <div className="gauge-container" style={{ border: '1px solid #ccc', width: '300px', height: '30px' }}>
        <div className="gauge" style={{
            width: `${lambda * 100}%`,
            height: '100%',
            backgroundColor: lambda > 0.95 ? 'green' : 'orange',
            transition: 'width 0.5s ease'
        }}>
            {lambda.toFixed(4)}
        </div>
      </div>
      <StatusLight status={lambda > 0.95 ? 'LOCKED' : 'DRIFT'} />
    </div>
  );
};

export default CoherenceMonitor;
