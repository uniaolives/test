import React from 'react';

const OrbVisualizer: React.FC = () => {
    return (
        <div>
            <h1>🜏 Arkhe Orb Visualizer</h1>
            <p>Projecting ℂ × ℝ³ × ℤ → ℝ⁴</p>
            <canvas id="orb-canvas" width="800" height="600" />
        </div>
    );
};

export default OrbVisualizer;
