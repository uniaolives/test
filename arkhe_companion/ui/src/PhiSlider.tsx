import React, { useState } from 'react';

export const PhiSlider: React.FC = () => {
    const [phi, setPhi] = useState(0.618);

    return (
        <div className="phi-knob">
            <label>φ-Knob: {phi.toFixed(3)}</label>
            <input
                type="range"
                min="0"
                max="1"
                step="0.001"
                value={phi}
                onChange={(e) => setPhi(parseFloat(e.target.value))}
            />
        </div>
    );
};
