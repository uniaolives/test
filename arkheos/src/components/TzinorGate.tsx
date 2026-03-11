import React from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../store';
import { Button, Card, Elevation, Icon, Tag } from '@blueprintjs/core';
import { sendObserveRequest } from '../services/websocket';

export const TzinorGate: React.FC = () => {
  const { orbs } = useSelector((state: RootState) => state.teknet);

  const handleObserve = (uqi: string) => {
    console.log(`[HTTP/4] Emitting OBSERVE for ${uqi}`);
    sendObserveRequest(uqi);
  };

  return (
    <div className="tzinor-gate">
      <h3 className="bp5-heading">
        <Icon icon="layout-skew-grid" /> TZINOR GATE (HTTP/4)
      </h3>
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
