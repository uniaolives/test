import { Dispatch } from '@reduxjs/toolkit';
import { updateState, setOrbs } from '../store/slices/teknetSlice';

let socket: WebSocket | null = null;

export const connectWebSocket = (dispatch: Dispatch) => {
  socket = new WebSocket('ws://localhost:8000/ws');

  socket.onopen = () => {
    console.log('Conectado ao backend Arkhe(n)');
  };

  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'ORB_DISCOVERY') {
      dispatch(setOrbs(data.orbs));
    } else {
      dispatch(updateState(data));
    }
  };

  socket.onerror = (error) => {
    console.error('WebSocket error:', error);
  };

  socket.onclose = () => {
    console.log('WebSocket fechado. Tentar reconectar em 3s...');
    setTimeout(() => connectWebSocket(dispatch), 3000);
  };
};

export const sendObserveRequest = (uqi: string) => {
  if (socket && socket.readyState === WebSocket.OPEN) {
    const request = {
      method: 'OBSERVE',
      uqi: uqi,
      headers: {
        origin: new Date().toISOString(),
        target: new Date().toISOString(),
        lambda_2: 0.99,
        confinement: 'INFINITE_WELL',
        paradox_policy: 'REJECT',
        mobius_twist: 0.0
      }
    };
    socket.send(JSON.stringify(request));
  }
};

export const sendEmitRequest = (uqi: string, lambda_2: number) => {
  if (socket && socket.readyState === WebSocket.OPEN) {
    const request = {
      method: 'EMIT',
      uqi: uqi,
      headers: {
        origin: new Date().toISOString(),
        target: new Date().toISOString(),
        lambda_2: lambda_2,
        confinement: 'FINITE_WELL',
        paradox_policy: 'REJECT',
        mobius_twist: 0.0
      },
      payload: [],
      // Em uma integração real, a assinatura GRAIL seria gerada aqui via HSM ou Wallet
      grail_signature: {
        signature: "U09NQVRJQ19TSUdOQVRVUkU=",
        rollout_id: "arkhe-mvo-01",
        timestamp: new Date().toISOString(),
        logic_hash: new Array(32).fill(0)
      }
    };
    socket.send(JSON.stringify(request));
  }
};

export const disconnectWebSocket = () => {
  if (socket) socket.close();
};
