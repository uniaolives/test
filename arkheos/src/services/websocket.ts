import { Dispatch } from '@reduxjs/toolkit';
import { updateState } from '../store/slices/teknetSlice';

let socket: WebSocket | null = null;

export const connectWebSocket = (dispatch: Dispatch) => {
  socket = new WebSocket('ws://localhost:8000/ws');

  socket.onopen = () => {
    console.log('Conectado ao backend Arkhe(n)');
  };

  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    dispatch(updateState(data));
  };

  socket.onerror = (error) => {
    console.error('WebSocket error:', error);
  };

  socket.onclose = () => {
    console.log('WebSocket fechado. Tentar reconectar em 3s...');
    setTimeout(() => connectWebSocket(dispatch), 3000);
  };
};

export const disconnectWebSocket = () => {
  if (socket) socket.close();
};
