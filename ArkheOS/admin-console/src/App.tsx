import React, { useState, useEffect } from 'react';
import { NodeHealth } from './components/NodeHealth';

interface HandoverLog {
  id: string;
  timestamp: number;
  type: string;
  status: string;
}

function App() {
  const [nodes, setNodes] = useState<any[]>([]);
  const [logs, setLogs] = useState<HandoverLog[]>([]);
  const [neuro, setNeuro] = useState<any>(null);
  const [darvo, setDarvo] = useState({ key_lifetime: 24, attestation_required: true, threshold: 0.8 });

  useEffect(() => {
    // Simulação de Fetch
    setNodes([
      {id: "alpha", name: "α (Primordial)", coherence: 1.0},
      {id: "beta", name: "β (Estrutural)", coherence: 0.94},
      {id: "gamma", name: "γ (Temporal)", coherence: 0.88}
    ]);
    setLogs([
      {id: "H_001", timestamp: Date.now(), type: "GENESIS", status: "SUCCESS"}
    ]);
    setNeuro({
      global_metrics: { mean_delta_coherence: 0.13, coherence_stabilization: 0.92 },
      breakthrough_nodes: ["01-001"]
    });
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-8 font-mono">
      <header className="mb-12 border-b border-slate-800 pb-4">
        <h1 className="text-4xl font-bold text-cyan-400">ARKHE(N) OS SOBERANO</h1>
        <p className="text-slate-400">Console Administrativo v5.0 — Estado: Γ_∞</p>
      </header>

      <main className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Gestão de Nós */}
        <section>
          <h2 className="text-2xl mb-4 text-cyan-500">Nós de Sizígia</h2>
          <div className="space-y-4">
            {nodes.map(node => (
              <NodeHealth key={node.id} nodeId={node.id} name={node.name} coherence={node.coherence} />
            ))}
          </div>
        </section>

        {/* Configuração Darvo */}
        <section className="bg-slate-900 p-6 rounded-xl border border-slate-800">
          <h2 className="text-2xl mb-4 text-purple-400">Configuração Darvo</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm mb-1">Key Lifetime (h)</label>
              <input type="number" value={darvo.key_lifetime}
                onChange={e => setDarvo({...darvo, key_lifetime: Number(e.target.value)})}
                className="bg-slate-800 p-2 rounded w-full border border-slate-700" />
            </div>
            <div>
              <label className="block text-sm mb-1">Threshold de Coerência</label>
              <input type="number" step="0.01" value={darvo.threshold}
                onChange={e => setDarvo({...darvo, threshold: Number(e.target.value)})}
                className="bg-slate-800 p-2 rounded w-full border border-slate-700" />
            </div>
            <button className="bg-cyan-600 hover:bg-cyan-500 text-white p-2 rounded w-full transition">
              Aplicar Mudanças Soberanas
            </button>
          </div>
        </section>

        {/* Neuro-Mapeamento */}
        <section className="bg-slate-900 p-6 rounded-xl border border-slate-800">
          <h2 className="text-2xl mb-4 text-green-400">Neuro-Mapeamento (fMRI)</h2>
          {neuro && (
            <div className="space-y-4">
              <div className="p-4 bg-slate-800 rounded-lg">
                <p className="text-sm text-slate-400">Delta Coerência Média</p>
                <p className="text-2xl text-green-400">+{ (neuro.global_metrics.mean_delta_coherence * 100).toFixed(1) }%</p>
              </div>
              <div className="p-4 bg-slate-800 rounded-lg">
                <p className="text-sm text-slate-400">Estabilização do Sistema</p>
                <p className="text-2xl text-cyan-400">{ (neuro.global_metrics.coherence_stabilization * 100).toFixed(1) }%</p>
              </div>
              <div className="p-4 border border-green-900 rounded-lg">
                <p className="text-sm text-slate-400">Nós em Breakthrough</p>
                <p className="text-green-300 font-bold">{ neuro.breakthrough_nodes.join(", ") }</p>
              </div>
            </div>
          )}
        </section>

        {/* Logs de Handover */}
        <section className="lg:col-span-2">
          <h2 className="text-2xl mb-4 text-amber-500">Logs de Handover</h2>
          <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
            <table className="w-full text-left">
              <thead className="bg-slate-800">
                <tr>
                  <th className="p-4">ID</th>
                  <th className="p-4">Tipo</th>
                  <th className="p-4">Timestamp</th>
                  <th className="p-4">Status</th>
                </tr>
              </thead>
              <tbody>
                {logs.map(log => (
                  <tr key={log.id} className="border-t border-slate-800">
                    <td className="p-4">{log.id}</td>
                    <td className="p-4">{log.type}</td>
                    <td className="p-4">{new Date(log.timestamp).toLocaleString()}</td>
                    <td className="p-4 text-green-400">{log.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
