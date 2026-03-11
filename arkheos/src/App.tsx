import React, { useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from './store/index';
import { ArkheSphere } from './components/ArkheSphere';
import { ControlPanel } from './components/ControlPanel';
import { TzinorGate } from './components/TzinorGate';
import { connectWebSocket, disconnectWebSocket } from './services/websocket';
import '@blueprintjs/core/lib/css/blueprint.css';
import '@blueprintjs/icons/lib/css/blueprint-icons.css';
import './App.css';

const App: React.FC = () => {
  const dispatch = useDispatch();
  const { q_value, state } = useSelector((state: RootState) => state.teknet);

  useEffect(() => {
    connectWebSocket(dispatch);
    return () => disconnectWebSocket();
  }, [dispatch]);

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>ARKHE(N) OS v1.0</h1>
        <div className="system-status">
          STATE: <span className={`status-${state.toLowerCase()}`}>{state}</span>
        </div>
      </header>
      <main className="app-main">
        <div className="visualizer-container">
          <ArkheSphere coherence={q_value} />
        </div>
        <div className="controls-overlay">
          <ControlPanel />
          <TzinorGate />
        </div>
      </main>
    </div>
  );
};

export default App;
