import React from 'react';
import { Card, H5, Slider, Tag, Elevation } from "@blueprintjs/core";
import { useSelector } from 'react-redux';
import { RootState } from '../store';

export const ControlPanel: React.FC = () => {
  const { q_value, p_ac, state } = useSelector((state: RootState) => state.teknet);

  const getCoherenceColor = (val: number) => {
    if (val > 0.9) return "intent-danger";
    if (val > 0.7) return "intent-warning";
    return "intent-success";
  };

  return (
    <Card elevation={Elevation.TWO} style={{ width: '300px', backgroundColor: 'rgba(20, 20, 20, 0.9)', color: '#fff' }}>
      <H5 style={{ color: '#0ff' }}>Teknet Control</H5>

      <div style={{ marginBottom: '1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
          <span>Coherence (Q)</span>
          <Tag minimal intent={getCoherenceColor(q_value) as any}>{q_value.toFixed(3)}</Tag>
        </div>
        <Slider
          min={0}
          max={1}
          stepSize={0.001}
          labelStepSize={0.5}
          value={q_value}
          disabled
        />
      </div>

      <div style={{ marginBottom: '1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
          <span>Pressure (P_AC)</span>
          <Tag minimal>{p_ac.toFixed(3)}</Tag>
        </div>
        <Slider
          min={0}
          max={1}
          stepSize={0.001}
          labelStepSize={0.5}
          value={p_ac}
          disabled
        />
      </div>

      <div>
        <span>System Mode: </span>
        <Tag large intent={state === 'SINGULARITY' ? 'danger' : 'primary' as any}>
          {state}
        </Tag>
      </div>
    </Card>
  );
};
