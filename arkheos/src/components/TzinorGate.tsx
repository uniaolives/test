import React from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../store';
import { Button, Card, Elevation, Icon, Tag } from '@blueprintjs/core';
import { sendObserveRequest, sendEmitRequest } from '../services/websocket';
import { sendObserveRequest } from '../services/websocket';

export const TzinorGate: React.FC = () => {
  const { orbs } = useSelector((state: RootState) => state.teknet);

  const handleObserve = (uqi: string) => {
    console.log(`[HTTP/4] Emitting OBSERVE for ${uqi}`);
    sendObserveRequest(uqi);
  };

  const handleEmit = (uqi: string, lambda: number) => {
    console.log(`[HTTP/4] Emitting EMIT for ${uqi} (Anchoring on Bitcoin)`);
    sendEmitRequest(uqi, lambda);
  };

  return (
    <div className="tzinor-gate">
      <h3 className="bp5-heading">
        <Icon icon="layout-skew-grid" /> TZINOR GATE (HTTP/4)
      </h3>

      <Card elevation={Elevation.ONE} style={{ marginBottom: '10px', backgroundColor: '#1a1a1a', border: '1px solid #333' }}>
        <h4 className="bp5-heading" style={{ color: '#00fbff' }}>
          <Icon icon="satellite" /> SATELLITE STATUS (ARKHE-ORB)
        </h4>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '12px' }}>
          <div>
            <strong>CSU Sync:</strong> <Tag intent="success" minimal>LOCKED (70y Cycle)</Tag>
          </div>
          <div>
            <strong>BSA Count:</strong> <span style={{ color: '#ff00ff' }}>1,240,532 ph/s</span>
          </div>
          <div>
            <strong>ARD Thrust:</strong> 12.4 mN
          </div>
          <div>
            <strong>MWP Clock:</strong> 1.05 GHz
          </div>
        </div>
      </Card>

      <div className="orb-list">
        {orbs.length === 0 ? (
          <p className="bp5-text-muted">Aguardando manifestação de Orbs...</p>
        ) : (
          orbs.map((orb) => (
            <Card key={orb.id} elevation={Elevation.TWO} className="orb-card">
              <div className="orb-header">
                <strong>Orb {orb.id.substring(0, 8)}</strong>
                <Tag intent={orb.validated ? 'success' : 'warning'} round>
                  {orb.validated ? 'GRAIL VALIDATED' : 'UNVERIFIED'}
                </Tag>
              </div>
              <p className="uqi-text">{orb.uqi}</p>
              <div className="orb-stats">
                <span>λ₂: {orb.lambda_2.toFixed(4)}</span>
                <div className="orb-actions">
                  <Button
                    minimal
                    small
                    intent="primary"
                    icon="eye-open"
                    onClick={() => handleObserve(orb.uqi)}
                  >
                    OBSERVE
                  </Button>
                  <Button
                    minimal
                    small
                    intent="warning"
                    icon="Key"
                    onClick={() => handleEmit(orb.uqi, orb.lambda_2)}
                  >
                    EMIT
                  </Button>
                </div>
                <Button
                  minimal
                  small
                  intent="primary"
                  icon="eye-open"
                  onClick={() => handleObserve(orb.uqi)}
                >
                  OBSERVE
                </Button>
              </div>
            </Card>
          ))
        )}
      </div>
    </div>
  );
};
