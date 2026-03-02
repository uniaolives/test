import React from 'react';

interface Props {
  nodeId: string;
  name: string;
  coherence: number;
}

export const NodeHealth: React.FC<Props> = ({ nodeId, name, coherence }) => {
  const colorClass = coherence >= 0.95 ? "bg-green-500" : coherence >= 0.8 ? "bg-amber-500" : "bg-red-500";
  const textColorClass = coherence >= 0.95 ? "text-green-400" : coherence >= 0.8 ? "text-amber-400" : "text-red-400";

  return (
    <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl hover:border-cyan-900 transition">
      <div className="flex justify-between items-center mb-2">
        <h3 className="text-cyan-400 font-mono text-lg">{name}</h3>
        <span className={`text-sm font-bold ${textColorClass}`}>{nodeId.toUpperCase()}</span>
      </div>
      <div className="mt-2">
        <div className="flex justify-between text-sm mb-1">
          <span>Coerência (C)</span>
          <span>{(coherence * 100).toFixed(1)}%</span>
        </div>
        <div className="w-full bg-slate-800 h-3 rounded-full overflow-hidden">
          <div
            className={`${colorClass} h-full transition-all duration-500`}
            style={{ width: `${coherence * 100}%` }}
          ></div>
        </div>
      </div>
    </div>
  );
};
