// src/context/OrchContext.tsx
// Context API para estado do Orch-Core (interface biológica)

import React, { createContext, useContext, useState, useCallback } from 'react';

interface OrchState {
  connected: boolean;
  coherenceTime: number; // ms
  neuralActivity: Float32Array | null;
  quantumSignature: string | null;
  lastSync: number | null;
}

interface OrchContextType extends OrchState {
  connect: () => Promise<void>;
  disconnect: () => void;
  calibrate: () => Promise<void>;
  measureCoherence: () => Promise<number>;
}

const parseOrchPacket = (value: Uint8Array): any => {
    return { type: 'COHERENCE', coherenceMs: 100, activityVector: new Float32Array(10) };
}

const computeSignature = (lambda: number): string => {
    return `sig-${lambda}`;
}

const OrchContext = createContext<OrchContextType | null>(null);

export const OrchProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, setState] = useState<OrchState>({
    connected: false,
    coherenceTime: 0,
    neuralActivity: null,
    quantumSignature: null,
    lastSync: null
  });

  const connect = useCallback(async () => {
    // Conecta ao Orch-Core via WebUSB/WebSerial
    const device = await (navigator as any).serial.requestPort({
      filters: [{ usbVendorId: 0x7f3b }] // Vendor ID Arkhe(n)
    });

    await device.open({ baudRate: 115200 });

    // Handshake com Totem
    const writer = device.writable.getWriter();
    const encoder = new TextEncoder();
    await writer.write(encoder.encode('TOTEM:7f3b49c8...\n'));
    writer.releaseLock();

    setState(s => ({ ...s, connected: true }));

    // Inicia loop de leitura
    const reader = device.readable.getReader();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      // Parse de pacotes Orch-Core
      const packet = parseOrchPacket(value);
      if (packet.type === 'COHERENCE') {
        setState(s => ({
          ...s,
          coherenceTime: packet.coherenceMs,
          neuralActivity: packet.activityVector,
          lastSync: Date.now()
        }));
      }
    }
  }, []);

  const measureCoherence = useCallback(async () => {
    if (!state.connected) throw new Error('Orch-Core desconectado');

    // Solicita medição ao hardware
    // Retorna λ_sync local
    const response = await fetch('/api/orch/measure', {
      method: 'POST',
      body: JSON.stringify({ timestamp: Date.now() })
    });

    const { lambdaSync } = await response.json();
    setState(s => ({ ...s, quantumSignature: computeSignature(lambdaSync) }));

    return lambdaSync;
  }, [state.connected]);

  return (
    <OrchContext.Provider value={{
      ...state,
      connect,
      disconnect: () => setState(s => ({ ...s, connected: false })),
      calibrate: async () => { /* ... */ },
      measureCoherence
    }}>
      {children}
    </OrchContext.Provider>
  );
};

export const useOrch = () => {
  const ctx = useContext(OrchContext);
  if (!ctx) throw new Error('useOrch must be used within OrchProvider');
  return ctx;
};
